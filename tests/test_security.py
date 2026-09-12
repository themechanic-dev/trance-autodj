"""Passwords, encryption at rest, and masking."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from app.core.security import (
    SecretStore,
    generate_session_secret,
    hash_password,
    load_or_create_key,
    mask,
    verify_password,
)


def test_password_round_trip():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


def test_verify_never_raises_on_rubbish():
    assert not verify_password("x", "not-a-bcrypt-hash")
    assert not verify_password("", "")


def test_empty_password_is_refused():
    with pytest.raises(ValueError):
        hash_password("")


def test_key_file_is_owner_only(tmp_path: Path):
    key_path = tmp_path / "master.key"
    key = load_or_create_key(key_path)
    assert key_path.is_file()
    mode = stat.S_IMODE(key_path.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {mode:o}"
    # Loading again returns the same key rather than rotating it.
    assert load_or_create_key(key_path) == key


def test_secret_store_round_trip(tmp_path: Path):
    store = SecretStore.open(tmp_path / "secrets.enc")
    store.set("youtube_stream_key", "abcd-efgh-ijkl-mnop")
    assert store.get("youtube_stream_key") == "abcd-efgh-ijkl-mnop"
    assert store.get("missing", "fallback") == "fallback"
    assert store.names() == ["youtube_stream_key"]


def test_secret_store_file_is_not_plaintext(tmp_path: Path):
    path = tmp_path / "secrets.enc"
    store = SecretStore.open(path)
    store.set("youtube_stream_key", "abcd-efgh-ijkl-mnop")
    raw = path.read_bytes()
    assert b"abcd-efgh-ijkl-mnop" not in raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_secret_store_survives_a_reopen(tmp_path: Path):
    path = tmp_path / "secrets.enc"
    SecretStore.open(path).set("k", "value-to-persist")
    assert SecretStore.open(path).get("k") == "value-to-persist"


def test_deleting_a_secret_removes_it(tmp_path: Path):
    store = SecretStore.open(tmp_path / "secrets.enc")
    store.set("k", "v-something")
    store.delete("k")
    assert store.get("k") == ""
    assert store.names() == []


def test_wrong_master_key_does_not_crash(tmp_path: Path):
    """A rotated key must degrade to 'secrets unreadable', not to a traceback."""
    path = tmp_path / "secrets.enc"
    SecretStore.open(path, key_path=tmp_path / "a.key").set("k", "v-something")
    other = SecretStore.open(path, key_path=tmp_path / "b.key")
    assert other.get("k") == ""


def test_session_secret_is_stable(tmp_path: Path):
    path = tmp_path / "session.key"
    first = generate_session_secret(path)
    assert first
    assert generate_session_secret(path) == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), ("abcd", "••••"), ("abcd-efgh-ijkl-mnop", "•" * 15 + "mnop")],
)
def test_mask(value: str, expected: str):
    assert mask(value) == expected
