"""FastAPI entry point: the dashboard and every service it supervises.

One process owns the audio stack and the broadcast so that the dashboard can
start and stop them. The alternative — separate containers plus the Docker
socket — would be more moving parts and a far larger blast radius for no gain
on a single-operator system.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api import audio as audio_api
from app.api import logs as logs_api
from app.api import settings_api
from app.api import stream as stream_api
from app.api import system as system_api
from app.api import visuals as visuals_api
from app.core.auth import (
    FAILED_LOGIN_DELAY_S,
    MIN_PASSWORD_LENGTH,
    SESSION_COOKIE,
    AuthState,
    authenticated,
    is_public,
)
from app.core.db import dispose, session_scope
from app.core.logging import get_logger
from app.core.runtime import SECRET_PASSWORD_HASH, build_runtime
from app.services.audio import library, playlists, playlog
from app.services.audio.manager import AudioManager
from app.services.music import queue as music_queue
from app.services.stream.manager import StreamManager
from app.services.system import settings as settings_service
from app.services.system.preflight import Status, run_preflight

log = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"
SERVICE_NAME = "web"


def _resume(app: FastAPI, runtime, audio) -> None:
    """Come back up the way we went down.

    The container restarts itself on failure and on boot; without this it
    would come back silent and off air, reporting perfect health, until
    somebody opened a browser. Only intent is restored — a stack that was
    deliberately stopped stays stopped.
    """
    if not runtime.config.app.resume_on_start:
        return

    if audio.was_on_air():
        log.info("the audio was on air when we stopped; bringing it back")
        with contextlib.suppress(Exception):
            audio.start()

    if not app.state.stream.wanted_live():
        return

    if runtime.has_stream_key():
        log.info("the broadcast was live when we stopped; going back on")
        with contextlib.suppress(Exception):
            app.state.stream.start()
        return

    # And stop claiming to want it. Leaving the intent set means the state
    # file goes on saying "live" for a station that has no destination, and
    # the next key entered anywhere would put it on air without anybody
    # pressing anything.
    log.warning("the broadcast was live when we stopped, but there is no stream key")
    app.state.stream.state.want_live = False
    app.state.stream._write_state()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runtime = build_runtime(service=SERVICE_NAME)

    # Fold in whatever the dashboard has changed since the file was written.
    with contextlib.suppress(Exception), session_scope() as session:
        runtime.config = settings_service.apply(session)

    app.state.runtime = runtime

    auth_state = AuthState.from_config(runtime.config)
    if not auth_state.password_hash:
        # The password chosen through /setup lives in the encrypted store, not
        # in config.yaml. Without this it would be forgotten on every restart
        # and the setup page would come back.
        auth_state.password_hash = runtime.secrets.get(SECRET_PASSWORD_HASH)
    app.state.auth = auth_state

    report = run_preflight(runtime.config, runtime.paths)
    app.state.preflight = report
    for check in report.checks:
        if check.status is Status.FAIL:
            log.error("preflight: %s — %s", check.title, check.detail)
        elif check.status is Status.WARN:
            log.warning("preflight: %s — %s", check.title, check.detail)
    log.info(
        "preflight complete",
        extra={"ok": report.ok, "failures": len(report.failures), "warnings": len(report.warnings)},
    )

    audio = AudioManager(runtime)
    app.state.audio = audio
    app.state.stream = StreamManager(runtime)

    with contextlib.suppress(Exception):
        # There must always be something to play and a script to run before
        # anyone presses start.
        with session_scope() as session:
            library.scan(session, runtime.paths)
            playlists.ensure_default(session, runtime.paths, runtime.config)
        audio.write_configuration()

    # The log has to be complete on a machine nobody is watching, which is
    # every machine this runs on once it works, so it is a thread of our own
    # rather than something hung off a request.
    app.state.playlog = playlog.Recorder(runtime.paths)
    app.state.playlog.start()

    # Composing runs from a queue in the database, one niced process at a
    # time, so a batch of a hundred survives a restart and never outranks
    # the broadcast for the CPU.
    app.state.composer = music_queue.ComposerWorker(runtime)
    app.state.composer.start()

    _resume(app, runtime, audio)

    try:
        yield
    finally:
        log.info("shutting down")
        # The broadcast goes first: closing the RTMP session cleanly matters
        # more than anything else here, and it needs the audio still running.
        # remember=False: we are going away, not being told to go off air.
        # Recording "stopped" here is what made resume_on_start do nothing at
        # all — every clean shutdown erased the very intent it needed.
        with contextlib.suppress(Exception):
            app.state.composer.stop()
        with contextlib.suppress(Exception):
            app.state.playlog.stop()
        with contextlib.suppress(Exception):
            app.state.stream.stop(remember=False)
        with contextlib.suppress(Exception):
            audio.stop(remember=False)
        dispose()


def create_app() -> FastAPI:  # noqa: PLR0915 - one statement per route registration
    app = FastAPI(
        title="Trance AutoDJ & Visual Broadcaster",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    for router in (
        system_api.router,
        visuals_api.router,
        audio_api.router,
        stream_api.router,
        settings_api.router,
        logs_api.router,
    ):
        app.include_router(router)

    static_dir = WEB_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Asked for by every browser regardless of what the page declares —
        # on the login page, on a 404, on a bare GET — so it has to answer.
        return FileResponse(static_dir / "img" / "favicon.ico", media_type="image/x-icon")

    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.state.templates = templates

    # ---- authentication -------------------------------------------------

    @app.middleware("http")
    async def require_login(request: Request, call_next):  # type: ignore[no-untyped-def]
        auth: AuthState = request.app.state.auth
        path = request.url.path

        if auth.enabled and not auth.configured and not is_public(path):
            # First run: nothing is protected yet because nothing has been
            # chosen. Send a browser to the setup page rather than a login
            # form it cannot possibly pass — but answer an API client with a
            # status code and a sentence, not a redirect it cannot follow.
            if path.startswith("/api/"):
                return JSONResponse(
                    {"detail": "no dashboard password has been set yet; open /setup"},
                    status_code=401,
                )
            return RedirectResponse("/setup", status_code=303)

        if not is_public(path) and not authenticated(request):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "not authenticated"}, status_code=401)
            return RedirectResponse("/login", status_code=303)

        return await call_next(request)

    # ---- pages ----------------------------------------------------------

    def page(name: str, active: str):  # type: ignore[no-untyped-def]
        def render(request: Request):  # type: ignore[no-untyped-def]
            return templates.TemplateResponse(
                request=request,
                name=name,
                context={"version": __version__, "active_page": active},
            )

        return render

    app.get("/", include_in_schema=False)(page("dashboard.html", "home"))
    app.get("/audio", include_in_schema=False)(page("audio.html", "audio"))
    app.get("/visuals", include_in_schema=False)(page("visuals.html", "visuals"))
    app.get("/stream", include_in_schema=False)(page("stream.html", "stream"))
    app.get("/settings", include_in_schema=False)(page("settings.html", "settings"))
    app.get("/logs", include_in_schema=False)(page("logs.html", "logs"))

    @app.get("/preflight", include_in_schema=False)
    def preflight_page(request: Request, refresh: bool = False):  # type: ignore[no-untyped-def]
        runtime = request.app.state.runtime
        report = getattr(request.app.state, "preflight", None)
        if report is None or refresh:
            report = run_preflight(runtime.config, runtime.paths)
            request.app.state.preflight = report
        return templates.TemplateResponse(
            request=request,
            name="preflight.html",
            context={
                "version": __version__,
                "active_page": "preflight",
                "report": report,
                "grouped": report.by_category(),
                "config": runtime.config,
                "data_dir": str(runtime.paths.root),
            },
        )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ---- login ----------------------------------------------------------

    @app.get("/login", include_in_schema=False)
    def login_form(request: Request):  # type: ignore[no-untyped-def]
        if not request.app.state.auth.configured:
            return RedirectResponse("/setup", status_code=303)
        return templates.TemplateResponse(
            request=request, name="login.html", context={"version": __version__, "error": ""}
        )

    @app.post("/login", include_in_schema=False)
    async def login(  # type: ignore[no-untyped-def]
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        import asyncio

        auth: AuthState = request.app.state.auth
        if not auth.check(username, password):
            # Same delay whatever was wrong, so timing says nothing.
            await asyncio.sleep(FAILED_LOGIN_DELAY_S)
            log.warning("failed login attempt", extra={"username": username})
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"version": __version__, "error": "Wrong username or password."},
                status_code=401,
            )

        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            auth.issue(),
            max_age=auth.max_age_s,
            httponly=True,
            samesite="lax",
            secure=auth.cookie_secure,
        )
        log.info("signed in")
        return response

    @app.post("/logout", include_in_schema=False)
    def logout() -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.get("/setup", include_in_schema=False)
    def setup_form(request: Request):  # type: ignore[no-untyped-def]
        if request.app.state.auth.configured:
            return RedirectResponse("/login", status_code=303)
        return templates.TemplateResponse(
            request=request, name="setup.html", context={"version": __version__, "error": ""}
        )

    @app.post("/setup", include_in_schema=False)
    def setup(  # type: ignore[no-untyped-def]
        request: Request,
        password: str = Form(...),
        confirm: str = Form(...),
    ):
        auth: AuthState = request.app.state.auth
        if auth.configured:
            raise HTTPException(409, "a password is already set")

        error = ""
        if password != confirm:
            error = "The passwords do not match."
        elif len(password) < MIN_PASSWORD_LENGTH:
            error = f"Use at least {MIN_PASSWORD_LENGTH} characters."
        if error:
            return templates.TemplateResponse(
                request=request,
                name="setup.html",
                context={"version": __version__, "error": error},
                status_code=400,
            )

        auth.set_password(password)
        # Persisted next to the other secrets, not in config.yaml, so the
        # hash never lands in a file anyone is likely to copy around.
        request.app.state.runtime.secrets.set(SECRET_PASSWORD_HASH, auth.password_hash)
        log.info("dashboard password set")

        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            auth.issue(),
            max_age=auth.max_age_s,
            httponly=True,
            samesite="lax",
            secure=auth.cookie_secure,
        )
        return response

    @app.exception_handler(500)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error — see the logs"})

    return app


app = create_app()


def main() -> None:
    """Console entry point: ``trance-autodj``."""
    import uvicorn

    from app.core.config import load_config

    cfg = load_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        log_config=None,  # our own logging is already configured
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
