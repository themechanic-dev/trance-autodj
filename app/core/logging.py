"""Structured logging with rotation and unconditional secret redaction.

Two formats: ``json`` for the file and for production stdout, ``console`` for
a human reading along. Both pass through the same redaction filter, so there
is no configuration in which a secret can reach a log.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.core.config import Config

# Values registered here are replaced wherever they appear in a log record.
_SECRETS: set[str] = set()

# Belt-and-braces: even an unregistered secret is caught when it appears next
# to one of these words. Ordered so the longest key names match first.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A labelled value: "password: hunter2", "stream_key=abcd", "token = x".
    # An explicit ":" or "=" is required — matching a bare space after the
    # word turns "no password is set" into "no password ***REDACTED*** set".
    re.compile(
        r"(?i)\b(stream[-_ ]?key|password|passwd|secret|api[-_ ]?key|token|authorization)"
        r"\b\s*[:=]\s*(?:Bearer\s+)?(?P<value>[^\s,;'\"}\])]+)"
    ),
    # A YouTube stream key: four to five dash-separated alphanumeric groups.
    re.compile(r"\b(?P<value>[a-z0-9]{4}(?:-[a-z0-9]{4}){3,5})\b", re.IGNORECASE),
    # rtmp://host/live2/<key>
    re.compile(r"(?P<prefix>rtmps?://[^\s/]+/[^\s/]+/)(?P<value>[^\s'\"]+)"),
)

REDACTED = "***REDACTED***"

# Below this length a "secret" is too short to replace safely: blanking a
# three-character string would corrupt ordinary words in every log line.
MIN_SECRET_LENGTH = 6

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


def register_secret(value: str | None) -> None:
    """Mark a literal value as never-loggable.

    Call this the moment a secret is read from anywhere. Short values are
    ignored: redacting a two-character string would mangle ordinary text.
    """
    if value and len(value) >= MIN_SECRET_LENGTH:
        _SECRETS.add(value)


def redact(text: str) -> str:
    for secret in _SECRETS:
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        text = pattern.sub(_replace_value_group, text)
    return text


def _replace_value_group(match: re.Match[str]) -> str:
    whole = match.group(0)
    value = match.group("value")
    if not value or value == REDACTED:
        return whole
    return whole.replace(value, REDACTED)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        for key, value in list(record.__dict__.items()):
            if key not in _RESERVED and isinstance(value, str):
                record.__dict__[key] = redact(value)
        return True


class JsonFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "service": getattr(record, "service", _service_name()),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key == "service":
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                payload[key] = repr(value)
            else:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s | %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.color:
            prefix = self._COLORS.get(record.levelname, "")
            if prefix:
                text = f"{prefix}{text}{self._RESET}"
        return text


def _service_name() -> str:
    return os.environ.get("TAD_SERVICE", "app")


def setup_logging(cfg: Config, log_dir: Path, *, service: str | None = None) -> None:
    """Install handlers on the root logger. Safe to call more than once."""
    if service:
        os.environ["TAD_SERVICE"] = service

    level = getattr(logging, cfg.app.log_level.upper())
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    redaction = RedactionFilter()

    stream_handler = logging.StreamHandler(sys.stderr)
    if cfg.app.log_format == "json":
        stream_handler.setFormatter(JsonFormatter())
    else:
        stream_handler.setFormatter(ConsoleFormatter(color=sys.stderr.isatty()))
    stream_handler.addFilter(redaction)
    root.addHandler(stream_handler)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{_service_name()}.log",
            maxBytes=cfg.app.log_max_bytes,
            backupCount=cfg.app.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(redaction)
        root.addHandler(file_handler)
    except OSError as exc:
        # A read-only or missing data dir must not stop the process from
        # starting: preflight will report it far more clearly than a traceback.
        root.warning("file logging disabled: %s", exc)

    # These are chatty at INFO and say nothing we need.
    for noisy in ("uvicorn.access", "multipart", "PIL", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def register_config_secrets(cfg: Config, extra: Iterable[str] = ()) -> None:
    """Register every secret-bearing config value in one call."""
    for value in (
        cfg.stream.youtube.stream_key,
        cfg.auth.session_secret,
        cfg.auth.password_hash,
        cfg.audio.icecast.source_password,
        cfg.audio.icecast.admin_password,
        *extra,
    ):
        register_secret(value)


__all__ = [
    "REDACTED",
    "get_logger",
    "redact",
    "register_config_secrets",
    "register_secret",
    "setup_logging",
]
