"""Single-user authentication for the dashboard.

Homelab-grade on purpose: one password, a signed session cookie, no user
management. But not open either — whoever reaches this port can broadcast to
the channel and delete the library, so it is closed by default and the first
visit makes you choose a password.
"""

from __future__ import annotations

import hmac
import time
from dataclasses import dataclass

from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.core.config import Config
from app.core.logging import get_logger
from app.core.security import hash_password, verify_password

log = get_logger(__name__)

SESSION_COOKIE = "tad_session"
SALT = "trance-autodj-session"

# Paths that must work before anyone has logged in.
PUBLIC_PATHS = frozenset(
    {"/login", "/health", "/api/system/health", "/static", "/favicon.ico", "/setup"}
)

MIN_PASSWORD_LENGTH = 8

# A brute-force delay. Not a lockout — locking out the only user of a homelab
# box is worse than the attack it prevents — but enough to make guessing slow.
FAILED_LOGIN_DELAY_S = 1.0


@dataclass
class AuthState:
    """Runtime authentication state, held on the app."""

    enabled: bool
    username: str
    password_hash: str
    serializer: URLSafeTimedSerializer
    max_age_s: int
    cookie_secure: bool

    @property
    def configured(self) -> bool:
        return bool(self.password_hash)

    @classmethod
    def from_config(cls, cfg: Config) -> AuthState:
        return cls(
            enabled=cfg.auth.enabled,
            username=cfg.auth.username,
            password_hash=cfg.auth.password_hash,
            serializer=URLSafeTimedSerializer(cfg.auth.session_secret, salt=SALT),
            max_age_s=cfg.auth.session_max_age_s,
            cookie_secure=cfg.auth.cookie_secure,
        )

    # -- passwords ---------------------------------------------------------

    def set_password(self, password: str) -> None:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"use at least {MIN_PASSWORD_LENGTH} characters")
        self.password_hash = hash_password(password)

    def check(self, username: str, password: str) -> bool:
        # compare_digest on the username too, so a wrong username and a wrong
        # password take the same time.
        user_ok = hmac.compare_digest(username.encode(), self.username.encode())
        password_ok = verify_password(password, self.password_hash)
        return user_ok and password_ok

    # -- sessions ----------------------------------------------------------

    def issue(self) -> str:
        return self.serializer.dumps({"u": self.username, "t": int(time.time())})

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        try:
            data = self.serializer.loads(token, max_age=self.max_age_s)
        except (BadSignature, SignatureExpired):
            return False
        return isinstance(data, dict) and data.get("u") == self.username


def is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS)


def authenticated(request: Request) -> bool:
    auth: AuthState | None = getattr(request.app.state, "auth", None)
    if auth is None or not auth.enabled:
        return True
    return auth.valid(request.cookies.get(SESSION_COOKIE))


__all__ = [
    "FAILED_LOGIN_DELAY_S",
    "MIN_PASSWORD_LENGTH",
    "PUBLIC_PATHS",
    "SESSION_COOKIE",
    "AuthState",
    "authenticated",
    "is_public",
]
