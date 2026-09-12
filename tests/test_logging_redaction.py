"""A secret must not survive a trip through the logging stack."""

from __future__ import annotations

import json
import logging

from app.core.logging import (
    REDACTED,
    JsonFormatter,
    RedactionFilter,
    redact,
    register_secret,
)


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, msg, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_registered_secret_is_replaced():
    register_secret("super-secret-value")
    assert "super-secret-value" not in redact("key is super-secret-value here")


def test_short_values_are_not_registered():
    """Redacting a 3-character string would mangle ordinary prose."""
    register_secret("abc")
    assert redact("abc def") == "abc def"


def test_stream_key_shape_is_caught_without_registration():
    assert "a1b2-c3d4-e5f6-g7h8" not in redact("using a1b2-c3d4-e5f6-g7h8 now")


def test_rtmp_url_key_is_stripped():
    out = redact("rtmp://a.rtmp.youtube.com/live2/zzzz-yyyy-xxxx-wwww")
    assert "zzzz-yyyy-xxxx-wwww" not in out
    assert "rtmp://a.rtmp.youtube.com/live2/" in out


def test_labelled_values_are_caught():
    for text in (
        "password: hunter2000",
        "stream_key=abcdefghijkl",
        "Authorization: Bearer sometokenvalue",
    ):
        assert REDACTED in redact(text)


def test_filter_scrubs_message_args_and_extras():
    register_secret("another-secret-value")
    record = _record("value=%s", stream_key="another-secret-value")
    record.args = ("another-secret-value",)
    RedactionFilter().filter(record)
    rendered = JsonFormatter().format(record)
    assert "another-secret-value" not in rendered
    assert json.loads(rendered)["stream_key"] == REDACTED


def test_json_formatter_emits_parseable_lines():
    record = _record("hello", block_id="abc123")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello"
    assert payload["level"] == "info"
    assert payload["block_id"] == "abc123"
    assert payload["ts"].endswith("Z")


def test_ordinary_prose_survives():
    """Redaction must not mangle the messages it passes over.

    An earlier pattern accepted a bare space after the keyword, which turned
    the preflight line "no password is set" into "no password *** set" — a
    log that actively misleads the person reading it.
    """
    for text in (
        "no password is set",
        "the stream key is not configured yet",
        "secret store opened",
        "token bucket refilled",
    ):
        assert redact(text) == text


def test_bearer_tokens_lose_the_token_not_the_word():
    out = redact("Authorization: Bearer abcdef123456")
    assert "abcdef123456" not in out
    assert "Bearer" in out
