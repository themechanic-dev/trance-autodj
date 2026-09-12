"""Render the Liquidsoap and Icecast configuration files.

Both are generated from Jinja2 templates and written into the data directory,
never edited by hand. That is what lets the dashboard change the crossfade
length or the loudness target and have it take effect on the next restart —
and it means the running configuration always matches config.yaml.
"""

from __future__ import annotations

import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.core.config import Config
from app.core.logging import get_logger
from app.core.paths import Paths

log = get_logger(__name__)

# Liquidsoap log levels: 1 critical .. 5 debug.
_LIQ_LOG_LEVEL = {"error": 1, "warning": 2, "info": 3, "debug": 4}
# Icecast: 1 error .. 4 debug.
_ICE_LOG_LEVEL = {"error": 1, "warning": 2, "info": 3, "debug": 4}

ICECAST_BASEDIR = "/usr/share/icecast2"
ICECAST_MAX_CLIENTS = 20
ICECAST_BURST_SIZE = 65535


def liq_float(value: float) -> str:
    """Format a number as a Liquidsoap float literal.

    Liquidsoap needs a decimal point, and `f"{v}."` is not it: for -14.0 that
    produces "-14.0.", which fails to parse with an error that points at line
    one and tells you nothing. Always go through here.
    """
    number = float(value)
    if number == int(number):
        return f"{int(number)}.0"
    return f"{number:.6f}".rstrip("0")


def _environment(config_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(config_dir / "templates")),
        undefined=StrictUndefined,  # a missing variable must fail loudly
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 - these are config files, not HTML
    )


def render_liquidsoap(
    cfg: Config,
    paths: Paths,
    *,
    source_password: str,
    station_name: str = "Trance AutoDJ",
    station_description: str = "24/7 trance with generative visuals",
) -> str:
    audio = cfg.audio
    crossfade = audio.crossfade
    env = _environment(Path(cfg.app.config_dir))
    return env.get_template("autodj.liq.j2").render(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        log_level=_LIQ_LOG_LEVEL.get(cfg.app.log_level, 3),
        telnet_host=audio.liquidsoap.telnet_host,
        telnet_port=audio.liquidsoap.telnet_port,
        # In a rotation the file order *is* the play order: this application
        # decided which list follows which, so Liquidsoap must read straight
        # through. Any shuffling wanted inside a list is done when the file is
        # written, where the list boundaries are still known.
        mode=(
            "normal"
            if audio.playlist_playback == "rotation"
            else {"shuffle": "randomize", "sequential": "normal", "random": "random"}[audio.mode]
        ),
        playlist_file=str(paths.active_playlist),
        nowplaying_dir=str(paths.nowplaying_file.parent),
        safe_track=audio.safe_track,
        loudness_enabled=audio.loudness.enabled,
        target_lufs=liq_float(audio.loudness.target_lufs),
        duration=liq_float(crossfade.duration_s),
        fade_in=liq_float(crossfade.duration_s),
        fade_out=liq_float(crossfade.duration_s),
        width=liq_float(crossfade.width_s),
        minimum_track=liq_float(crossfade.minimum_track_s),
        fade_in_curve=crossfade.fade_in_curve,
        fade_out_curve=crossfade.fade_out_curve,
        beat_aligned=crossfade.beat_aligned,
        beat_align_cue_in=crossfade.beat_aligned and crossfade.beat_align_cue_in,
        nowplaying_file=str(paths.nowplaying_file),
        audio_format=audio.icecast.format,
        bitrate_k=audio.icecast.bitrate_k,
        icecast_host=audio.icecast.host,
        icecast_port=audio.icecast.port,
        icecast_password=source_password,
        icecast_mount=audio.icecast.mount.lstrip("/"),
        station_name=station_name,
        station_description=station_description,
        retry_delay=liq_float(audio.retry_delay_s),
        extra_config=audio.liquidsoap.extra_config,
    )


def render_icecast(
    cfg: Config,
    paths: Paths,
    *,
    source_password: str,
    admin_password: str,
) -> str:
    env = _environment(Path(cfg.app.config_dir))
    return env.get_template("icecast.xml.j2").render(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        location="localhost",
        admin_email="admin@localhost",
        max_clients=ICECAST_MAX_CLIENTS,
        burst_size=ICECAST_BURST_SIZE,
        source_password=source_password,
        admin_password=admin_password,
        hostname=cfg.audio.icecast.host,
        port=cfg.audio.icecast.port,
        bind_address=cfg.audio.icecast.host,
        basedir=ICECAST_BASEDIR,
        logdir=str(paths.logs),
        webroot=f"{ICECAST_BASEDIR}/web",
        adminroot=f"{ICECAST_BASEDIR}/admin",
        pidfile=str(paths.state / "icecast.pid"),
        loglevel=_ICE_LOG_LEVEL.get(cfg.app.log_level, 3),
    )


def write_configs(
    cfg: Config,
    paths: Paths,
    *,
    source_password: str,
    admin_password: str,
) -> tuple[Path, Path]:
    """Write both files and return their paths."""
    paths.state.mkdir(parents=True, exist_ok=True)

    liq = render_liquidsoap(cfg, paths, source_password=source_password)
    paths.liquidsoap_script.write_text(liq, encoding="utf-8")

    ice = render_icecast(cfg, paths, source_password=source_password, admin_password=admin_password)
    paths.icecast_config.write_text(ice, encoding="utf-8")
    # The file holds the source password in plain text; nobody else needs it.
    paths.icecast_config.chmod(0o600)

    log.info(
        "audio configuration written",
        extra={
            "liquidsoap": str(paths.liquidsoap_script),
            "icecast": str(paths.icecast_config),
        },
    )
    return paths.liquidsoap_script, paths.icecast_config


__all__ = ["liq_float", "render_icecast", "render_liquidsoap", "write_configs"]
