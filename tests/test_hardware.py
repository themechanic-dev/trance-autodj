"""Capability detection, especially the 'auto' resolution rules."""

from __future__ import annotations

import pytest

from app.core import hardware
from app.core.hardware import GpuInfo, resolve_ai_backend, resolve_encoder
from app.core.proc import CommandResult


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in (
        hardware.gpu_info,
        hardware.torch_cuda_available,
        hardware.openvino_available,
        hardware.nvenc_usable,
        hardware.ffmpeg_encoders,
        hardware.ffmpeg_filters,
        hardware.ffmpeg_muxers,
        hardware.ffmpeg_version,
    ):
        fn.cache_clear()
    yield


def test_cpu_count_is_at_least_one():
    assert hardware.cpu_count() >= 1


def test_load_average_has_three_values():
    assert len(hardware.load_average()) == 3


def test_explicit_libx264_is_never_second_guessed(monkeypatch):
    monkeypatch.setattr(hardware, "nvenc_usable", lambda *_a, **_k: (True, "available"))
    encoder, reason = resolve_encoder("libx264")
    assert encoder == "libx264"
    assert "explicit" in reason


def test_auto_picks_nvenc_when_it_actually_works(monkeypatch):
    monkeypatch.setattr(hardware, "nvenc_usable", lambda *_a, **_k: (True, "verified"))
    encoder, reason = resolve_encoder("auto")
    assert encoder == "h264_nvenc"
    assert "verified" in reason


def test_auto_falls_back_to_x264_and_says_why(monkeypatch):
    monkeypatch.setattr(
        hardware, "nvenc_usable", lambda *_a, **_k: (False, "no NVIDIA GPU detected")
    )
    encoder, reason = resolve_encoder("auto")
    assert encoder == "libx264"
    assert "no NVIDIA GPU detected" in reason


def test_requesting_nvenc_without_a_gpu_degrades_instead_of_failing(monkeypatch):
    """A config written for a GPU host must still boot on a CPU-only VM."""
    monkeypatch.setattr(hardware, "nvenc_usable", lambda *_a, **_k: (False, "card removed"))
    encoder, reason = resolve_encoder("h264_nvenc")
    assert encoder == "libx264"
    assert "card removed" in reason


def test_ai_backend_none_is_respected():
    assert resolve_ai_backend("none")[0] == "none"


def test_ai_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(hardware, "gpu_info", lambda: GpuInfo(present=True, name="RTX 3060"))
    monkeypatch.setattr(hardware, "torch_cuda_available", lambda: True)
    backend, reason = resolve_ai_backend("auto")
    assert backend == "cuda"
    assert "RTX 3060" in reason


def test_ai_auto_falls_through_to_sdcpp(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "gpu_info", lambda: GpuInfo(present=False))
    monkeypatch.setattr(hardware, "torch_cuda_available", lambda: False)
    fake = tmp_path / "sd"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    backend, _ = resolve_ai_backend("auto", sdcpp_binary="sd")
    assert backend == "sdcpp"


def test_ai_auto_reports_none_with_all_the_reasons(monkeypatch):
    monkeypatch.setattr(hardware, "gpu_info", lambda: GpuInfo(present=False))
    monkeypatch.setattr(hardware, "torch_cuda_available", lambda: False)
    monkeypatch.setattr(hardware, "openvino_available", lambda: False)
    monkeypatch.setenv("PATH", "")
    backend, reason = resolve_ai_backend("auto")
    assert backend == "none"
    for expected in ("cuda:", "sdcpp:", "openvino:"):
        assert expected in reason


def test_cpu_count_respects_a_cgroup_quota(monkeypatch, tmp_path):
    """Inside a container limited to 2 cores we must not plan for 16."""
    quota = tmp_path / "cpu.max"
    quota.write_text("200000 100000")
    real_is_file = hardware.Path.is_file

    def fake_is_file(self):
        return True if str(self) == "/sys/fs/cgroup/cpu.max" else real_is_file(self)

    def fake_read_text(self, *a, **k):
        if str(self) == "/sys/fs/cgroup/cpu.max":
            return "200000 100000"
        return quota.read_text()

    monkeypatch.setattr(hardware.Path, "is_file", fake_is_file)
    monkeypatch.setattr(hardware.Path, "read_text", fake_read_text)
    assert hardware.cpu_count() == 2


# --- ffmpeg listing parser -------------------------------------------------
#
# Verbatim excerpts from ffmpeg 8.0.1. The three listings differ in ways that
# matter: only two of them have a separator rule, and the flag column is 6, 3
# and 2 characters wide respectively.

ENCODERS_OUTPUT = """Encoders:
 V..... = Video
 A..... = Audio
 .....D = Supports direct rendering method 1
 ------
 V....D a64multi             Multicolor charset for Commodore 64 (codec a64_multi)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""

FILTERS_OUTPUT = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  V = Video input/output
  | = Source or sink filter
 TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 TS displace          VVV->V     Displace pixels.
 T. vignette          V->V       Make or reverse a vignette effect.
 .S xfade             VV->V      Cross fade one video with another video.
 .. zoompan           V->V       Apply Zoom & Pan effect.
"""

MUXERS_OUTPUT = """Formats:
 D.. = Demuxing supported
 .E. = Muxing supported
 ..d = Is a device
 ---
  E  3g2             3GP2 (3GPP2 file format)
  E  mpegts          MPEG-TS (MPEG-2 Transport Stream)
  E  flv             FLV (Flash Video)
"""


@pytest.mark.parametrize(
    ("kind", "output", "expected", "not_expected"),
    [
        ("encoders", ENCODERS_OUTPUT, {"libx264", "aac", "a64multi"}, {"=", "------"}),
        (
            "filters",
            FILTERS_OUTPUT,
            {"xfade", "displace", "vignette", "zoompan", "aap"},
            {"=", "Audio", "Timeline"},
        ),
        ("muxers", MUXERS_OUTPUT, {"mpegts", "flv", "3g2"}, {"=", "---", "Demuxing"}),
    ],
)
def test_ffmpeg_listing_parser(monkeypatch, kind, output, expected, not_expected):
    """The filters listing has no separator rule; the parser must not need one."""
    monkeypatch.setattr(hardware, "which", lambda _b: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        hardware,
        "run",
        lambda _argv, _opts=None: CommandResult(
            argv=[], returncode=0, stdout=output, stderr="", duration_s=0.0
        ),
    )
    names = set(hardware._list_ffmpeg("ffmpeg", kind))
    assert expected <= names, f"missing {expected - names}"
    assert not (not_expected & names), f"legend text leaked in: {not_expected & names}"


def test_parser_returns_empty_when_ffmpeg_is_absent(monkeypatch):
    monkeypatch.setattr(hardware, "which", lambda _b: None)
    assert hardware._list_ffmpeg("ffmpeg", "filters") == ()
