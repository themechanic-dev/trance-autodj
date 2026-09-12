"""Encode profile and block validation — the guard on "-c:v copy"."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Config, load_config
from app.services.visual.encoder import EncodeProfile, validate_block


def profile_for(**video: object) -> EncodeProfile:
    cfg = Config.model_validate({"video": video} if video else {})
    return EncodeProfile._build(cfg, "libx264")


def test_block_args_carry_every_parameter_that_must_not_vary():
    args = " ".join(profile_for().block_args())
    for expected in (
        "-c:v libx264",
        "-pix_fmt yuv420p",
        "-r 30",
        "-s 1280x720",
        "-profile:v high",
        "-level 4.1",
        "-g 60",
        "-keyint_min 60",
        "-bf 2",
    ):
        assert expected in args, expected


def test_x264_disables_scene_cut_detection():
    """Scene cuts would place keyframes by content, so two blocks would no
    longer share a GOP structure and the copied joins would break."""
    assert "-sc_threshold" in profile_for().block_args()


def test_nvenc_uses_its_own_spelling():
    cfg = Config()
    nvenc = EncodeProfile._build(cfg, "h264_nvenc")
    args = " ".join(nvenc.block_args())
    assert "-c:v h264_nvenc" in args
    assert "-no-scenecut" in args
    # -sc_threshold does not exist on nvenc and makes ffmpeg exit.
    assert "-sc_threshold" not in args


def test_intermediate_encode_is_cheap():
    args = " ".join(profile_for().intermediate_args())
    assert "ultrafast" in args
    assert "libx264" in args


def test_level_is_scaled_the_way_ffprobe_reports_it():
    assert profile_for().level_int == 41


def test_fingerprint_tracks_the_parameters_that_matter():
    base = profile_for().fingerprint
    assert profile_for().fingerprint == base
    assert profile_for(gop=120, keyint_min=120).fingerprint != base
    assert profile_for(width=1920, height=1080).fingerprint != base


def test_profile_is_cached_per_settings(tmp_path: Path):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path)})
    assert EncodeProfile.from_config(cfg) is EncodeProfile.from_config(cfg)


# --- validation ----------------------------------------------------------


def fake_probe(monkeypatch, payload: dict) -> None:
    import app.services.visual.encoder as module

    monkeypatch.setattr(module, "probe", lambda *_a, **_k: payload)


def good_payload(duration: float = 600.0) -> dict:
    return {
        "format": {"duration": str(duration)},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
                "profile": "High",
                "level": 41,
            }
        ],
    }


@pytest.fixture
def block_file(tmp_path: Path) -> Path:
    path = tmp_path / "block.ts"
    path.write_bytes(b"x" * 1024)
    return path


def test_a_matching_block_passes(monkeypatch, block_file: Path):
    fake_probe(monkeypatch, good_payload())
    ok, problems = validate_block(block_file, profile_for(), expected_duration_s=600.0)
    assert ok, problems


def test_missing_file_is_rejected(tmp_path: Path):
    ok, problems = validate_block(tmp_path / "nope.ts", profile_for())
    assert not ok
    assert "does not exist" in problems[0]


def test_empty_file_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.ts"
    path.touch()
    ok, problems = validate_block(path, profile_for())
    assert not ok
    assert "empty" in problems[0]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("codec_name", "hevc", "codec"),
        ("width", 1920, "resolution"),
        ("pix_fmt", "yuv444p", "pixel format"),
        ("r_frame_rate", "25/1", "frame rate"),
        ("profile", "Main", "H.264 profile"),
    ],
)
def test_each_mismatch_is_caught(monkeypatch, block_file: Path, field, value, expected):
    payload = good_payload()
    payload["streams"][0][field] = value
    fake_probe(monkeypatch, payload)
    ok, problems = validate_block(block_file, profile_for())
    assert not ok
    assert any(expected in p for p in problems), problems


def test_an_audio_track_is_rejected(monkeypatch, block_file: Path):
    """Blocks carry no audio; the streamer maps audio from Icecast instead."""
    payload = good_payload()
    payload["streams"].append({"codec_type": "audio", "codec_name": "aac"})
    fake_probe(monkeypatch, payload)
    ok, problems = validate_block(block_file, profile_for())
    assert not ok
    assert any("no audio" in p for p in problems)


def test_a_wrong_duration_is_caught(monkeypatch, block_file: Path):
    fake_probe(monkeypatch, good_payload(duration=120.0))
    ok, problems = validate_block(block_file, profile_for(), expected_duration_s=600.0)
    assert not ok
    assert any("duration" in p for p in problems)


def test_a_lower_level_is_allowed_but_a_higher_one_is_not(monkeypatch, block_file: Path):
    payload = good_payload()
    payload["streams"][0]["level"] = 31
    fake_probe(monkeypatch, payload)
    assert validate_block(block_file, profile_for())[0]

    payload["streams"][0]["level"] = 51
    fake_probe(monkeypatch, payload)
    ok, problems = validate_block(block_file, profile_for())
    assert not ok
    assert any("level" in p for p in problems)


def test_unreadable_file_is_a_rejection_not_a_crash(monkeypatch, block_file: Path):
    import app.services.visual.encoder as module

    def boom(*_a, **_k):
        raise ValueError("not json")

    monkeypatch.setattr(module, "probe", boom)
    ok, problems = validate_block(block_file, profile_for())
    assert not ok
    assert "ffprobe" in problems[0]
