"""The audio stack: Icecast, then Liquidsoap, then never silence.

One object owns both processes so that starting them in the right order,
regenerating their configuration when settings change, and reporting a single
"is the audio up" answer all live in one place.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Config
from app.core.logging import get_logger, register_secret
from app.core.paths import Paths
from app.core.runtime import (
    SECRET_ICECAST_ADMIN,
    SECRET_ICECAST_SOURCE,
    Runtime,
)
from app.services.audio import nowplaying, telnet
from app.services.audio.process import SupervisedProcess
from app.services.audio.templates import write_configs

log = get_logger(__name__)

# How long to wait for Icecast to accept connections before starting
# Liquidsoap. Liquidsoap retries anyway, but starting in order keeps the log
# free of alarming-looking connection failures on every boot.
ICECAST_READY_TIMEOUT_S = 15.0
ICECAST_POLL_S = 0.4


def _generated_password() -> str:
    return secrets.token_urlsafe(24)


@dataclass
class AudioStatus:
    icecast_running: bool
    liquidsoap_running: bool
    telnet_reachable: bool
    stream_reachable: bool
    now_playing: dict
    icecast: dict
    liquidsoap: dict

    @property
    def on_air(self) -> bool:
        return self.icecast_running and self.liquidsoap_running and self.stream_reachable

    def as_dict(self) -> dict[str, object]:
        return {
            "on_air": self.on_air,
            "icecast_running": self.icecast_running,
            "liquidsoap_running": self.liquidsoap_running,
            "telnet_reachable": self.telnet_reachable,
            "stream_reachable": self.stream_reachable,
            "now_playing": self.now_playing,
            "processes": {"icecast": self.icecast, "liquidsoap": self.liquidsoap},
        }


class AudioManager:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.cfg: Config = runtime.config
        self.paths: Paths = runtime.paths
        self._lock = threading.Lock()
        self._icecast: SupervisedProcess | None = None
        self._liquidsoap: SupervisedProcess | None = None

    # -- credentials -------------------------------------------------------

    def _passwords(self) -> tuple[str, str]:
        """Source and admin passwords, generated once and kept encrypted.

        Config can pin them, but the default is a random pair the operator
        never has to see: this Icecast only ever serves localhost.
        """
        store = self.runtime.secrets
        source = self.cfg.audio.icecast.source_password or store.get(SECRET_ICECAST_SOURCE)
        admin = self.cfg.audio.icecast.admin_password or store.get(SECRET_ICECAST_ADMIN)
        if not source:
            source = _generated_password()
            store.set(SECRET_ICECAST_SOURCE, source)
        if not admin:
            admin = _generated_password()
            store.set(SECRET_ICECAST_ADMIN, admin)
        register_secret(source)
        register_secret(admin)
        return source, admin

    # -- configuration -----------------------------------------------------

    def write_configuration(self) -> tuple[Path, Path]:
        source, admin = self._passwords()
        return write_configs(self.cfg, self.paths, source_password=source, admin_password=admin)

    # -- process control ---------------------------------------------------

    def _ensure_processes(self) -> None:
        """Create the two process objects, without ever discarding a live one.

        Replacing a SupervisedProcess that still has a child running orphans
        both the child and its supervisor thread: nothing can reach them any
        more, so stop() cannot stop them, while the fresh object starts a
        child that can never bind the port the orphan is still holding and
        restarts it for the lifetime of the container. Every press of "start"
        used to add one more of those loops.
        """
        icecast_argv = [self.cfg.audio.icecast.binary, "-c", str(self.paths.icecast_config)]
        # Liquidsoap caches its compiled script under /usr/share, which a
        # container running unprivileged cannot write. Left alone it prints a
        # permission error on every start — harmless, but it looks like a
        # fault in the dashboard's process log, and it also means the script
        # is recompiled from scratch every time.
        cache_dir = self.paths.state / "liquidsoap-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        liquidsoap_argv = [self.cfg.audio.liquidsoap.binary, str(self.paths.liquidsoap_script)]

        # A changed binary path is the one case that does need a new object.
        # Stop the old child first so it is never left running unreachable.
        if self._icecast is not None and list(self._icecast.argv) != icecast_argv:
            self._icecast.stop()
            self._icecast = None
        if self._liquidsoap is not None and list(self._liquidsoap.argv) != liquidsoap_argv:
            self._liquidsoap.stop()
            self._liquidsoap = None

        if self._icecast is None:
            self._icecast = SupervisedProcess(name="icecast", argv=icecast_argv)
        if self._liquidsoap is None:
            self._liquidsoap = SupervisedProcess(
                name="liquidsoap",
                argv=liquidsoap_argv,
                env={"LIQ_CACHE_USER_DIR": str(cache_dir)},
            )

    def start(self) -> None:
        with self._lock:
            self.write_configuration()
            self._ensure_processes()

            if self.cfg.audio.icecast.managed:
                assert self._icecast is not None
                self._icecast.start()
                self._wait_for_icecast()

            # Whatever was playing belonged to the last run. Until Liquidsoap
            # reaches the first track boundary and writes its own metadata —
            # which can be minutes on a trance station — the dashboard would
            # otherwise keep showing it, including tracks that have since been
            # deleted from the library.
            with contextlib.suppress(OSError):
                self.paths.nowplaying_file.unlink(missing_ok=True)

            assert self._liquidsoap is not None
            self._liquidsoap.start()
            self._remember(on_air=True)
            log.info("audio stack started")

    def _wait_for_icecast(self) -> bool:
        deadline = time.monotonic() + ICECAST_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.icecast_reachable():
                return True
            time.sleep(ICECAST_POLL_S)
        log.warning(
            "Icecast did not answer within %.0fs; starting Liquidsoap anyway "
            "(it will keep retrying)",
            ICECAST_READY_TIMEOUT_S,
        )
        return False

    def stop(self, *, remember: bool = True) -> None:
        """Take the audio off air.

        `remember=False` is the application shutting down rather than being
        told to stop. The difference is the whole point of the state file: a
        restart has to come back on air, and an operator's "stop" has to stay
        stopped.
        """
        with self._lock:
            # Liquidsoap first: it disconnects cleanly from a server that is
            # still there, which avoids an error in the Icecast log on every
            # shutdown.
            if self._liquidsoap is not None:
                self._liquidsoap.stop()
            if self._icecast is not None:
                self._icecast.stop()
            if remember:
                self._remember(on_air=False)
            log.info("audio stack stopped", extra={"remembered": remember})

    def restart(self) -> None:
        self.stop()
        self.start()

    # -- remembered intent -------------------------------------------------

    def _remember(self, *, on_air: bool) -> None:
        """Record whether the operator wants audio, so a restart can restore it.

        This is intent, not status: it changes only when someone asks for the
        audio to start or stop, never when a process dies. A crash must come
        back on air; a deliberate stop must stay off.
        """
        try:
            self.paths.state.mkdir(parents=True, exist_ok=True)
            temporary = self.paths.audio_state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"on_air": on_air}), encoding="utf-8")
            temporary.replace(self.paths.audio_state_file)
        except OSError as exc:
            log.warning("could not write the audio state file: %s", exc)

    def was_on_air(self) -> bool:
        """Was the audio wanted when this data directory was last used?"""
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(self.paths.audio_state_file.read_text(encoding="utf-8"))
            return bool(data.get("on_air"))
        return False

    def apply_settings(self) -> None:
        """Regenerate the configuration and restart Liquidsoap only.

        Crossfade length, loudness target and playback mode all live in the
        generated script, so they need a Liquidsoap restart — but not an
        Icecast one, which would drop the streamer's connection.
        """
        with self._lock:
            self.write_configuration()
            if self._liquidsoap is not None:
                self._liquidsoap.restart()
                log.info("Liquidsoap restarted with new settings")

    # -- reachability ------------------------------------------------------

    @staticmethod
    def _http_request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
        """Build a request, refusing anything that is not plain HTTP.

        These URLs come from our own configuration rather than from a user,
        but the configuration is editable, and ``urlopen`` will happily follow
        a ``file:`` URL and read a local file. Checking the scheme costs one
        line and removes the whole class of surprise; the noqa markers below
        record that the check has been done, since the linter cannot see it.
        """
        scheme = urllib.parse.urlparse(url).scheme
        if scheme not in {"http", "https"}:
            raise ValueError(f"refusing to open a {scheme!r} URL")
        return urllib.request.Request(url, headers=headers or {})  # noqa: S310

    def icecast_reachable(self, timeout_s: float = 2.0) -> bool:
        url = f"http://{self.cfg.audio.icecast.host}:{self.cfg.audio.icecast.port}/status-json.xsl"
        try:
            request = self._http_request(url)
            with urllib.request.urlopen(request, timeout=timeout_s):  # noqa: S310
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def stream_reachable(self, timeout_s: float = 2.0) -> bool:
        """Is there actually audio on the mount point?

        Icecast answering is not the same as Liquidsoap being connected to it,
        and the difference is exactly "silence on air".
        """
        try:
            request = self._http_request(self.cfg.audio.icecast.stream_url, {"Icy-MetaData": "0"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                return bool(response.read(1))
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def telnet_reachable(self) -> bool:
        return telnet.is_alive(
            self.cfg.audio.liquidsoap.telnet_host, self.cfg.audio.liquidsoap.telnet_port
        )

    def reload_playlist(self) -> bool:
        return telnet.reload_playlist(
            self.cfg.audio.liquidsoap.telnet_host, self.cfg.audio.liquidsoap.telnet_port
        )

    def skip_track(self) -> bool:
        return telnet.skip(
            self.cfg.audio.liquidsoap.telnet_host, self.cfg.audio.liquidsoap.telnet_port
        )

    # -- status ------------------------------------------------------------

    def status(self) -> AudioStatus:
        icecast_status = (
            self._icecast.status().as_dict()
            if self._icecast
            else {"name": "icecast", "running": False}
        )
        liquidsoap_status = (
            self._liquidsoap.status().as_dict()
            if self._liquidsoap
            else {"name": "liquidsoap", "running": False}
        )
        return AudioStatus(
            icecast_running=bool(self._icecast and self._icecast.running),
            liquidsoap_running=bool(self._liquidsoap and self._liquidsoap.running),
            telnet_reachable=self.telnet_reachable(),
            stream_reachable=self.stream_reachable(),
            now_playing=nowplaying.read(self.paths.nowplaying_file).as_dict(),
            icecast=icecast_status,
            liquidsoap=liquidsoap_status,
        )

    def logs(self, which: str, count: int = 100) -> list[str]:
        process = {"icecast": self._icecast, "liquidsoap": self._liquidsoap}.get(which)
        return process.tail(count) if process else []


__all__ = ["AudioManager", "AudioStatus"]
