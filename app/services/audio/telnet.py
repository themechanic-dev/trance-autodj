"""Liquidsoap's telnet control surface.

Used to reload the playlist and to ask what is on air without restarting the
broadcast. It speaks a trivial line protocol terminated by "END".
"""

from __future__ import annotations

import socket

from app.core.logging import get_logger

log = get_logger(__name__)

TERMINATOR = b"END\r\n"
DEFAULT_TIMEOUT_S = 5.0


class TelnetError(RuntimeError):
    pass


def command(host: str, port: int, text: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Send one command and return its reply."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(text.encode("utf-8") + b"\n")
            chunks: list[bytes] = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
                joined = b"".join(chunks)
                if joined.endswith(TERMINATOR) or joined.endswith(b"END\n"):
                    break
            sock.sendall(b"quit\n")
    except OSError as exc:
        raise TelnetError(f"could not talk to Liquidsoap on {host}:{port}: {exc}") from exc

    reply = b"".join(chunks).decode("utf-8", errors="replace")
    for suffix in ("END\r\n", "END\n"):
        if reply.endswith(suffix):
            reply = reply[: -len(suffix)]
            break
    return reply.strip()


def reload_playlist(host: str, port: int, playlist_id: str = "main") -> bool:
    """Ask Liquidsoap to re-read the playlist file.

    reload_mode="watch" already does this when the file changes, so this is a
    belt-and-braces nudge for filesystems where inotify does not fire — a bind
    mount from a Windows host, for instance.
    """
    try:
        command(host, port, f"{playlist_id}.reload")
        return True
    except TelnetError as exc:
        log.warning("playlist reload over telnet failed: %s", exc)
        return False


def skip(host: str, port: int, source_id: str = "radio") -> bool:
    try:
        command(host, port, f"{source_id}.skip")
        return True
    except TelnetError as exc:
        log.warning("skip failed: %s", exc)
        return False


def uptime(host: str, port: int) -> str:
    try:
        return command(host, port, "uptime")
    except TelnetError:
        return ""


def is_alive(host: str, port: int, *, timeout_s: float = 2.0) -> bool:
    try:
        command(host, port, "version", timeout_s=timeout_s)
        return True
    except TelnetError:
        return False


__all__ = ["TelnetError", "command", "is_alive", "reload_playlist", "skip", "uptime"]
