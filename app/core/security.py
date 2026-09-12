"""Password hashing, session keys, and encryption at rest for the stream key.

The YouTube stream key is the one genuinely sensitive value in this project:
whoever holds it can broadcast to the channel. It is therefore never written
to config.yaml, never returned by an API in full, and never logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

from app.core.logging import get_logger, register_secret

log = get_logger(__name__)

# Environment variable holding the master key. When absent, a key file is
# generated next to the encrypted store.
MASTER_KEY_ENV = "TAD_MASTER_KEY"
KEY_FILE_NAME = "master.key"

_OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR  # 0600
MIN_PASSWORD_LENGTH = 8


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    if not password or not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed hash is a failed login, not a crash.
        return False


# --------------------------------------------------------------------------
# key material
# --------------------------------------------------------------------------


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive permissions from the start rather than
    # chmod-ing after the fact, which leaves a window where it is readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _OWNER_ONLY)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    path.chmod(_OWNER_ONLY)


def load_or_create_key(path: Path, *, env_var: str | None = None) -> bytes:
    """Return a urlsafe-base64 Fernet key, from env or from a 0600 file."""
    if env_var:
        raw = os.environ.get(env_var, "").strip()
        if raw:
            key = _normalise_key(raw)
            register_secret(key.decode("ascii"))
            return key

    if path.is_file():
        key = path.read_bytes().strip()
        try:
            Fernet(key)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path} does not contain a valid key") from exc
        register_secret(key.decode("ascii"))
        return key

    key = Fernet.generate_key()
    _write_private(path, key)
    log.info("generated a new master key", extra={"path": str(path)})
    register_secret(key.decode("ascii"))
    return key


def _normalise_key(raw: str) -> bytes:
    """Accept either a real Fernet key or an arbitrary passphrase."""
    candidate = raw.encode("ascii", errors="ignore")
    try:
        Fernet(candidate)
        return candidate
    except (ValueError, TypeError):
        digest = hashlib.sha256(raw.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)


def generate_session_secret(path: Path) -> str:
    if path.is_file():
        value = path.read_text(encoding="ascii").strip()
        if value:
            register_secret(value)
            return value
    value = secrets.token_urlsafe(48)
    _write_private(path, value.encode("ascii"))
    register_secret(value)
    return value


# --------------------------------------------------------------------------
# encrypted secret store
# --------------------------------------------------------------------------


class SecretStore:
    """A tiny encrypted key/value file for secrets the UI can change.

    Backed by Fernet (AES-128-CBC + HMAC). The file is unreadable without the
    master key, and both the file and the key are mode 0600.
    """

    def __init__(self, path: Path, key: bytes) -> None:
        self._path = path
        self._fernet = Fernet(key)
        self._cache: dict[str, str] | None = None

    @classmethod
    def open(cls, secrets_path: Path, *, key_path: Path | None = None) -> SecretStore:
        key_path = key_path or secrets_path.parent / KEY_FILE_NAME
        key = load_or_create_key(key_path, env_var=MASTER_KEY_ENV)
        return cls(secrets_path, key)

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        if not self._path.is_file():
            self._cache = {}
            return self._cache
        try:
            plaintext = self._fernet.decrypt(self._path.read_bytes())
        except InvalidToken:
            log.error(
                "the secret store could not be decrypted; the master key does not "
                "match. Existing secrets are unreadable and must be re-entered.",
                extra={"path": str(self._path)},
            )
            self._cache = {}
            return self._cache
        data: Any = json.loads(plaintext.decode("utf-8"))
        if not isinstance(data, dict):
            self._cache = {}
            return self._cache
        self._cache = {str(k): str(v) for k, v in data.items()}
        for value in self._cache.values():
            register_secret(value)
        return self._cache

    def _save(self, data: dict[str, str]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        _write_private(self._path, self._fernet.encrypt(payload))
        self._cache = data

    def get(self, name: str, default: str = "") -> str:
        return self._load().get(name, default)

    def set(self, name: str, value: str) -> None:
        data = dict(self._load())
        if value:
            data[name] = value
            register_secret(value)
        else:
            data.pop(name, None)
        self._save(data)

    def delete(self, name: str) -> None:
        self.set(name, "")

    def names(self) -> list[str]:
        return sorted(self._load())


def mask(value: str, *, keep: int = 4) -> str:
    """Render a secret for display: last few characters only."""
    if not value:
        return ""
    if len(value) <= keep:
        return "•" * len(value)
    return "•" * (len(value) - keep) + value[-keep:]


def _cli() -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(prog="python -m app.core.security")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash-password", help="prompt for a password and print its bcrypt hash")
    sub.add_parser("generate-key", help="print a fresh master key for TAD_MASTER_KEY")

    args = parser.parse_args()
    if args.command == "hash-password":
        first = getpass.getpass("Password: ")
        second = getpass.getpass("Repeat:   ")
        if first != second:
            print("Passwords do not match.")
            return 1
        if len(first) < MIN_PASSWORD_LENGTH:
            print(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
            return 1
        print(hash_password(first))
        return 0
    print(Fernet.generate_key().decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(_cli())


__all__ = [
    "MASTER_KEY_ENV",
    "SecretStore",
    "generate_session_secret",
    "hash_password",
    "load_or_create_key",
    "mask",
    "verify_password",
]
