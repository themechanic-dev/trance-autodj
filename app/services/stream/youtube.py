"""Naming the broadcast on YouTube.

RTMP carries no title. What we push is video and audio to an ingest point;
the name, the description and everything else a viewer sees belong to a
*broadcast* resource inside YouTube, and the only way to write them is the
Data API. So this module exists for one job: once we are on air, find the
broadcast our stream just created and give it the name the operator chose.

Two decisions worth knowing about.

**The device flow, not a redirect.** A self-hosted station has no public
HTTPS callback and often no browser on the machine at all. Google's flow for
limited-input devices hands us a short code, the operator approves it on a
phone, and we get a refresh token. Nothing has to be reachable from outside.

**Nothing here may ever stop the broadcast.** A quota error, an expired
token, a network blip — the station keeps playing and the title simply stays
as it was. Every entry point returns a result rather than raising into the
streamer.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.core.logging import get_logger, register_secret

log = get_logger(__name__)

DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 - a URL, not a token
API_ROOT = "https://www.googleapis.com/youtube/v3"

# Enough to rename our own broadcast, and no more.
SCOPE = "https://www.googleapis.com/auth/youtube"

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

# A broadcast does not exist the instant ffmpeg connects: YouTube creates it
# when it has seen enough of the stream. These bound the wait.
BROADCAST_POLL_S = 5.0
BROADCAST_WAIT_S = 120.0

HTTP_TIMEOUT_S = 15.0


class YoutubeError(RuntimeError):
    """Something the operator can act on: bad credentials, quota, no stream."""


@dataclass(frozen=True)
class DeviceCode:
    device_code: str
    user_code: str
    verification_url: str
    interval_s: int
    expires_in_s: int

    def as_dict(self) -> dict[str, object]:
        return {
            "user_code": self.user_code,
            "verification_url": self.verification_url,
            "expires_in_s": self.expires_in_s,
        }


def _post(url: str, data: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(url, data=body, method="POST")  # noqa: S310 - constant https
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        try:
            return json.loads(detail)
        except ValueError:
            raise YoutubeError(f"{exc.code} from Google: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise YoutubeError(f"could not reach Google: {exc}") from exc


def _api(method: str, path: str, token: str, *, params: dict[str, str], body: Any = None) -> Any:
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(params)}"
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=payload, method=method)  # noqa: S310 - https
    request.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise YoutubeError(f"YouTube returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise YoutubeError(f"could not reach YouTube: {exc}") from exc


# -- the device flow -------------------------------------------------------


def begin_device_flow(client_id: str) -> DeviceCode:
    data = _post(DEVICE_CODE_URL, {"client_id": client_id, "scope": SCOPE})
    if "device_code" not in data:
        raise YoutubeError(str(data.get("error_description") or data.get("error") or data))
    return DeviceCode(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_url=data.get("verification_url", "https://www.google.com/device"),
        interval_s=int(data.get("interval", 5)),
        expires_in_s=int(data.get("expires_in", 1800)),
    )


def poll_device_flow(client_id: str, client_secret: str, device_code: str) -> str | None:
    """One poll. Returns the refresh token, or None while still waiting.

    `authorization_pending` is the normal answer until the operator approves,
    so it is not an error and must not be shown as one.
    """
    data = _post(
        TOKEN_URL,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "device_code": device_code,
            "grant_type": DEVICE_GRANT,
        },
    )
    if "refresh_token" in data:
        register_secret(data["refresh_token"])
        return str(data["refresh_token"])

    error = str(data.get("error", ""))
    if error in {"authorization_pending", "slow_down"}:
        return None
    raise YoutubeError(str(data.get("error_description") or error or data))


def access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    data = _post(
        TOKEN_URL,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    token = data.get("access_token")
    if not token:
        raise YoutubeError(str(data.get("error_description") or data.get("error") or data))
    register_secret(str(token))
    return str(token)


# -- naming the broadcast --------------------------------------------------


def render_title(template: str, *, timezone: str, when: datetime | None = None) -> str:
    """Substitute the time placeholders, in the station's own timezone.

    A broadcast started at 03:00 local should say 03:00, not whatever UTC
    happened to be, and the container's clock is UTC.
    """
    if not template:
        return ""
    try:
        zone = ZoneInfo(timezone)
    except Exception:
        log.warning("unknown timezone %r; using UTC for the title", timezone)
        zone = ZoneInfo("UTC")
    moment = (when or datetime.now(tz=zone)).astimezone(zone)
    return (
        template.replace("{date}", moment.strftime("%d/%m/%Y"))
        .replace("{time}", moment.strftime("%H:%M"))
        .replace("{datetime}", moment.strftime("%d/%m/%Y %H:%M"))
    )


def active_broadcast(token: str) -> dict[str, Any] | None:
    """The broadcast that is on air now, if there is one."""
    for status in ("active", "upcoming"):
        data = _api(
            "GET",
            "liveBroadcasts",
            token,
            params={"part": "id,snippet,status", "broadcastStatus": status, "maxResults": "5"},
        )
        items = data.get("items") or []
        if items:
            return items[0]
    return None


def rename(token: str, broadcast: dict[str, Any], title: str, description: str) -> None:
    """Give the broadcast its name.

    The API replaces the whole snippet, so anything we do not send is erased —
    the scheduled start time included, which YouTube then rejects. Start from
    what is there and change only the two fields we own.
    """
    snippet = dict(broadcast.get("snippet") or {})
    snippet["title"] = title[:100]
    if description:
        snippet["description"] = description[:5000]
    _api(
        "PUT",
        "liveBroadcasts",
        token,
        params={"part": "id,snippet"},
        body={"id": broadcast["id"], "snippet": snippet},
    )


def wait_for_broadcast(
    token: str, *, deadline_s: float = BROADCAST_WAIT_S
) -> dict[str, Any] | None:
    """YouTube creates the broadcast a little after ffmpeg connects."""
    give_up_at = time.monotonic() + deadline_s
    while time.monotonic() < give_up_at:
        found = active_broadcast(token)
        if found is not None:
            return found
        time.sleep(BROADCAST_POLL_S)
    return None


__all__ = [
    "DeviceCode",
    "YoutubeError",
    "access_token",
    "active_broadcast",
    "begin_device_flow",
    "poll_device_flow",
    "rename",
    "render_title",
    "wait_for_broadcast",
]
