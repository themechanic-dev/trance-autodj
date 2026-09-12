"""From a seed to a tagged MP3 sitting in the music folder.

The composer works in floating point and knows nothing about files; this is
the part that turns its output into something the library can scan, Liquidsoap
can play and a player can show a name for.

Encoding goes through ffmpeg rather than a Python encoder because ffmpeg is
already a hard dependency of the streamer — adding lame bindings to get an
MP3 would mean a second encoder to keep working on two architectures.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.services.music import composer
from app.services.music.synth import SAMPLE_RATE

log = get_logger(__name__)

#: Generated music lives in its own folder inside the library. It scans like
#: anything else, but it can be told apart at a glance — and deleted without
#: taking anybody's own files with it.
GENERATED_DIR = "generated"

#: Two halves of a name, picked by the seed. A library full of "Track 4471"
#: is unreadable at a glance, and the now-playing overlay deserves better.
_FIRST = (
    "Aurora",
    "Cascade",
    "Ember",
    "Halcyon",
    "Lucid",
    "Meridian",
    "Nova",
    "Obsidian",
    "Parallax",
    "Quartz",
    "Solstice",
    "Tundra",
    "Umbra",
    "Velvet",
    "Zenith",
    "Cobalt",
    "Drifting",
    "Endless",
    "Fathom",
    "Glacier",
    "Horizon",
    "Ionosphere",
    "Kelvin",
)
_SECOND = (
    "Ascent",
    "Bloom",
    "Current",
    "Descent",
    "Echo",
    "Field",
    "Gate",
    "Hollow",
    "Interval",
    "Journey",
    "Kinetic",
    "Lantern",
    "Mirage",
    "Northern",
    "Orbit",
    "Passage",
    "Quiet",
    "Return",
    "Signal",
    "Threshold",
    "Undertow",
    "Vantage",
)


@dataclass(frozen=True)
class GeneratedTrack:
    """What was made, so the caller can file it without re-deriving anything."""

    path: Path
    relpath: str
    title: str
    bpm: float
    key: str
    seed: int
    duration_s: float
    style: str = "uplifting"


def name_for(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return f"{_FIRST[rng.integers(len(_FIRST))]} {_SECOND[rng.integers(len(_SECOND))]}"


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Sixteen-bit stereo, which is all the MP3 encoder will keep anyway."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _encode(source: Path, target: Path, *, bitrate_k: int, tags: dict[str, str]) -> None:
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_k}k",
    ]
    for key, value in tags.items():
        argv += ["-metadata", f"{key}={value}"]
    argv.append(str(target))

    result = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0 or not target.is_file():
        raise RuntimeError(f"ffmpeg could not encode the track: {result.stderr.strip()[:400]}")


def generate(
    music_dir: Path,
    *,
    seed: int,
    minutes: float = 5.0,
    bitrate_k: int = 192,
    style: str = "mixed",
    work_dir: Path | None = None,
) -> GeneratedTrack:
    """Compose, render and encode one track. Returns where it landed.

    The intermediate WAV is written beside the MP3 rather than in /tmp: a
    five-minute stereo track is fifty megabytes, and on a NAS /tmp is often a
    small tmpfs in RAM. It is removed as soon as the encode succeeds.
    """
    target_dir = music_dir / GENERATED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or target_dir

    spec = composer.plan(seed, minutes=minutes, style=style)
    title = name_for(seed)
    log.info(
        "composing a track",
        extra={
            "seed": seed,
            "title": title,
            "style": spec.style.name,
            "bpm": spec.bpm,
            "key": spec.key_name,
        },
    )
    audio = composer.render(spec)

    stem = f"{seed:08d}-{title.lower().replace(' ', '-')}"
    scratch = work_dir / f".{stem}.wav"
    final = target_dir / f"{stem}.mp3"
    try:
        _write_wav(scratch, audio)
        _encode(
            scratch,
            final,
            bitrate_k=bitrate_k,
            tags={
                "title": title,
                "artist": "Trance AutoDJ",
                # The style goes in the album so a player groups the psy
                # together and the progressive together, and in the genre so
                # a library search for "psy" finds it.
                "album": f"Generated Sessions · {spec.style.name}",
                "genre": f"Trance / {spec.style.name}",
                "comment": (
                    f"{spec.style.name} · {spec.bpm:.0f} BPM · {spec.key_name} · seed {seed}"
                ),
                "TBPM": f"{spec.bpm:.0f}",
            },
        )
    finally:
        scratch.unlink(missing_ok=True)

    log.info("track written", extra={"file": final.name, "seconds": round(spec.duration_s, 1)})
    return GeneratedTrack(
        path=final,
        relpath=f"{GENERATED_DIR}/{final.name}",
        title=title,
        bpm=spec.bpm,
        key=spec.key_name,
        seed=seed,
        duration_s=spec.duration_s,
        style=spec.style.name,
    )


def main(argv: list[str] | None = None) -> int:
    """Compose one track as a process of its own.

    The queue runs each track this way rather than in a thread of the web
    server: a process can be niced below the broadcast, killed cleanly when
    the operator changes their mind, and cannot take the dashboard down with
    it if numpy ever misbehaves. It prints one JSON object on success, which
    is all the worker needs to file the result.
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="compose one trance track")
    parser.add_argument("--music-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--style", default="mixed")
    parser.add_argument("--bitrate", type=int, default=192)
    args = parser.parse_args(argv)

    made = generate(
        args.music_dir,
        seed=args.seed,
        minutes=args.minutes,
        bitrate_k=args.bitrate,
        style=args.style,
    )
    json.dump(
        {
            "relpath": made.relpath,
            "title": made.title,
            "bpm": made.bpm,
            "key": made.key,
            "seed": made.seed,
            "duration_s": made.duration_s,
            "style": made.style,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["GENERATED_DIR", "GeneratedTrack", "generate", "main", "name_for"]
