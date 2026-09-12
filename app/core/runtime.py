"""The process-wide runtime: config, paths, secrets, database.

Built once at startup by whichever entry point is running — the web app, the
generator or the streamer — so all three share the same construction order
and the same failure messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import Config, load_config
from app.core.db import create_tables, init_engine
from app.core.logging import get_logger, register_config_secrets, setup_logging
from app.core.paths import Paths
from app.core.security import SecretStore, generate_session_secret

log = get_logger(__name__)

# Names used inside the encrypted secret store.
# These are lookup names inside the encrypted store, not secrets themselves.
SECRET_STREAM_KEY = "youtube_stream_key"  # noqa: S105
SECRET_ICECAST_SOURCE = "icecast_source_password"  # noqa: S105
SECRET_ICECAST_ADMIN = "icecast_admin_password"  # noqa: S105
SECRET_PASSWORD_HASH = "dashboard_password_hash"  # noqa: S105
# The Google credentials that let us name the broadcast. The client pair is
# the operator's own, from their Google Cloud project; the refresh token is
# what the device flow gives back once they approve it on a phone. All three
# live in the encrypted store beside the stream key, and none is ever logged.
SECRET_YT_CLIENT_ID = "youtube_client_id"  # noqa: S105
SECRET_YT_CLIENT_SECRET = "youtube_client_secret"  # noqa: S105
SECRET_YT_REFRESH_TOKEN = "youtube_refresh_token"  # noqa: S105


@dataclass
class Runtime:
    config: Config
    paths: Paths
    secrets: SecretStore
    service: str

    def stream_key(self) -> str:
        """Environment first, then the encrypted store.

        An operator who pins the key via TAD_STREAM__YOUTUBE__STREAM_KEY keeps
        control of it; the dashboard cannot overrule that.
        """
        return self.config.stream.youtube.stream_key or self.secrets.get(SECRET_STREAM_KEY)

    def has_stream_key(self) -> bool:
        return bool(self.stream_key())


def build_runtime(
    *,
    service: str,
    config_path: str | Path | None = None,
    create_schema: bool = True,
) -> Runtime:
    cfg = load_config(config_path)
    paths = Paths.from_config(cfg)
    paths.ensure()

    setup_logging(cfg, paths.logs, service=service)
    register_config_secrets(cfg)

    secrets = SecretStore.open(paths.secrets_file)
    # Reading the store registers every stored value with the log redactor,
    # so a secret cannot be logged even before anything asks for it.
    secrets.names()

    if not cfg.auth.session_secret:
        cfg.auth.session_secret = generate_session_secret(paths.session_key_file)

    init_engine(paths.db_file)
    if create_schema:
        create_tables()

    log.info(
        "runtime ready",
        extra={
            "service": service,
            "data_dir": str(paths.root),
            "config": str(cfg.source_path) if cfg.source_path else "defaults+env",
        },
    )
    return Runtime(config=cfg, paths=paths, secrets=secrets, service=service)


__all__ = [
    "SECRET_ICECAST_ADMIN",
    "SECRET_ICECAST_SOURCE",
    "SECRET_STREAM_KEY",
    "SECRET_YT_CLIENT_ID",
    "SECRET_YT_CLIENT_SECRET",
    "SECRET_YT_REFRESH_TOKEN",
    "Runtime",
    "build_runtime",
]
