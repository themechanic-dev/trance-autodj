"""Settings the dashboard is allowed to change, stored in the database.

Precedence, lowest to highest:

    defaults  ->  config.yaml  ->  these overrides  ->  TAD_* environment

The environment stays on top on purpose. An operator who pins a value in the
container's environment has made a deployment decision, and a web UI should
not be able to quietly undo it — so the UI shows such fields as locked instead
of pretending to change them.

Only the paths in EDITABLE can be set from here. Everything else — the data
directory, binary paths, the bind address — is deployment, not preference.
"""

from __future__ import annotations

import json
from typing import Any

from sqlmodel import Session, select

from app.core.config import ENV_PREFIX, Config, env_overrides, load_config
from app.core.logging import get_logger
from app.models.entities import Setting, utcnow

log = get_logger(__name__)

#: Dotted paths the dashboard may write, with a human label and a group.
EDITABLE: dict[str, tuple[str, str]] = {
    # audio
    "audio.mode": ("Playback order", "audio"),
    "audio.playlist_playback": ("Play one list or the rotation", "audio"),
    "audio.crossfade.duration_s": ("Crossfade length (s)", "audio"),
    "audio.crossfade.fade_in_curve": ("Fade-in curve", "audio"),
    "audio.crossfade.fade_out_curve": ("Fade-out curve", "audio"),
    "audio.crossfade.minimum_track_s": ("Skip crossfade below (s)", "audio"),
    "audio.crossfade.width_s": ("Loudness window (s)", "audio"),
    # Beat alignment is the reason the library has a BPM column and an
    # "Analyse tempo" button at all. Without these two here it could only be
    # switched on with an environment variable, which in a container means
    # editing the compose file and recreating everything.
    "audio.crossfade.beat_aligned": ("Align crossfade to whole bars", "audio"),
    "audio.crossfade.beat_align_cue_in": ("Skip to the first beat (trims audio)", "audio"),
    "audio.crossfade.max_cue_in_s": ("Most to trim (s)", "audio"),
    "audio.loudness.enabled": ("Normalise loudness", "audio"),
    "audio.loudness.target_lufs": ("Loudness target (LUFS)", "audio"),
    "audio.icecast.bitrate_k": ("Icecast bitrate (kbps)", "audio"),
    # The tempo search window. It is not decoration: a library whose tempo
    # sits outside it cannot be measured, and psytrance regularly runs past
    # the trance-shaped default of 152.
    "audio.analysis.min_bpm": ("Slowest tempo to look for", "audio"),
    "audio.analysis.max_bpm": ("Fastest tempo to look for", "audio"),
    # video profile
    "video.width": ("Width", "video"),
    "video.height": ("Height", "video"),
    "video.fps": ("Frame rate", "video"),
    "video.bitrate_k": ("Video bitrate (kbps)", "video"),
    "video.encoder": ("Encoder", "video"),
    "video.preset": ("x264 preset", "video"),
    # visuals
    "visual.clip.min_duration_s": ("Clip minimum (s)", "visuals"),
    "visual.clip.max_duration_s": ("Clip maximum (s)", "visuals"),
    "visual.clip.film_grain": ("Film grain", "visuals"),
    "visual.clip.vignette": ("Vignette", "visuals"),
    "visual.block.duration_s": ("Block length (s)", "visuals"),
    "visual.block.transition_min_s": ("Transition minimum (s)", "visuals"),
    "visual.block.transition_max_s": ("Transition maximum (s)", "visuals"),
    "visual.pool.min_blocks": ("Keep at least", "visuals"),
    "visual.pool.max_blocks": ("Stop above", "visuals"),
    "visual.pool.max_disk_gb": ("Disk budget (GB)", "visuals"),
    "visual.sources.ai_ratio": ("AI share of blocks", "visuals"),
    "visual.ai.backend": ("AI backend", "visuals"),
    "visual.ai.steps": ("Diffusion steps", "visuals"),
    "visual.procedural.render_scale": ("Render scale", "visuals"),
    # stream
    "stream.video_mode": ("Video mode", "stream"),
    "stream.audio.bitrate_k": ("Audio bitrate (kbps)", "stream"),
    "stream.youtube.rtmp_url": ("RTMP ingest URL", "stream"),
    # The name each broadcast gets. {date}, {time} and {datetime} are
    # filled in at the moment it goes live.
    "stream.youtube.title_template": ("Broadcast title", "stream"),
    "stream.youtube.description_template": ("Broadcast description", "stream"),
    "stream.play_count_weight": ("Prefer unplayed blocks", "stream"),
    "stream.watchdog.enabled": ("Restart automatically", "stream"),
    # cpu
    "cpu.load_threshold_factor": ("Load threshold (× cores)", "cpu"),
    "cpu.pause_during_stream": ("Pause while streaming", "cpu"),
    "cpu.quiet_hours.enabled": ("Quiet hours", "cpu"),
    "cpu.quiet_hours.start": ("Quiet hours start", "cpu"),
    "cpu.quiet_hours.end": ("Quiet hours end", "cpu"),
}

#: Changing one of these means the generated Liquidsoap script has changed.
AUDIO_RESTART_PREFIXES = ("audio.",)


def _nest(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    result: dict[str, Any] = {}
    cursor = result
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return result


def _deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_overrides(session: Session) -> dict[str, Any]:
    """Every stored override, as a flat {dotted path: value} mapping."""
    result: dict[str, Any] = {}
    for row in session.exec(select(Setting)).all():
        if row.key not in EDITABLE:
            continue
        try:
            result[row.key] = json.loads(row.value_json)
        except ValueError:
            continue
    return result


def environment_locked() -> set[str]:
    """Paths pinned by a TAD_* variable, which the UI must show as read-only."""
    locked: set[str] = set()
    flat = env_overrides()

    def walk(node: dict[str, Any], prefix: str = "") -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{path}.")
            else:
                locked.add(path)

    walk(flat)
    return {path for path in locked if path in EDITABLE}


def apply(session: Session, config_path: str | None = None) -> Config:
    """Rebuild the configuration with the stored overrides folded in."""
    overrides = read_overrides(session)
    if not overrides:
        return load_config(config_path)

    nested: dict[str, Any] = {}
    for path, value in overrides.items():
        nested = _deep_merge(nested, _nest(path, value))

    base = load_config(config_path).model_dump(exclude={"source_path"})
    merged = _deep_merge(base, nested)
    # The environment goes on last so a pinned value always wins.
    merged = _deep_merge(merged, {k: v for k, v in env_overrides().items() if k in merged})
    return Config.model_validate(merged)


def set_many(session: Session, values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate and store. Returns (accepted paths, rejected messages)."""
    accepted: list[str] = []
    rejected: list[str] = []
    locked = environment_locked()

    # Validate the whole set at once by building a candidate config: a value
    # that would make the configuration invalid must never reach the database.
    candidate: dict[str, Any] = {}
    for path, value in values.items():
        if path not in EDITABLE:
            rejected.append(f"{path}: not an editable setting")
            continue
        if path in locked:
            rejected.append(f"{path}: pinned by {ENV_PREFIX}{path.upper().replace('.', '__')}")
            continue
        candidate = _deep_merge(candidate, _nest(path, value))
        accepted.append(path)

    if not accepted:
        return accepted, rejected

    base = apply(session).model_dump(exclude={"source_path"})
    try:
        Config.model_validate(_deep_merge(base, candidate))
    except Exception as exc:
        return [], [*rejected, str(exc)]

    for path in accepted:
        row = session.get(Setting, path)
        if row is None:
            row = Setting(key=path)
        row.value_json = json.dumps(values[path])
        row.updated_at = utcnow()
        session.add(row)

    log.info("settings updated", extra={"changed": accepted})
    return accepted, rejected


def reset(session: Session, path: str) -> bool:
    row = session.get(Setting, path)
    if row is None:
        return False
    session.delete(row)
    return True


def describe(session: Session, cfg: Config) -> list[dict[str, Any]]:
    """Everything the settings page needs: value, label, group, lock state."""
    overrides = read_overrides(session)
    locked = environment_locked()
    current = cfg.model_dump(exclude={"source_path"})

    rows: list[dict[str, Any]] = []
    for path, (label, group) in EDITABLE.items():
        value: Any = current
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        rows.append(
            {
                "path": path,
                "label": label,
                "group": group,
                "value": value,
                "type": type(value).__name__,
                "overridden": path in overrides,
                "locked": path in locked,
            }
        )
    return rows


def needs_audio_restart(paths: list[str]) -> bool:
    return any(p.startswith(AUDIO_RESTART_PREFIXES) for p in paths)


__all__ = [
    "EDITABLE",
    "apply",
    "describe",
    "environment_locked",
    "needs_audio_restart",
    "read_overrides",
    "reset",
    "set_many",
]
