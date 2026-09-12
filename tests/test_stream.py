"""The broadcast command, the feeder, and now-playing state."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.paths import Paths
from app.services.audio import nowplaying
from app.services.stream.streamer import StreamHealth, build_command
from app.services.visual.encoder import EncodeProfile


def setup(tmp_path: Path, **environ: str):
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data"), **environ})
    paths = Paths.from_config(cfg)
    paths.ensure()
    return cfg, paths, EncodeProfile._build(cfg, "libx264")


def test_copy_mode_never_re_encodes_video(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert "-c:v" in argv
    assert argv[argv.index("-c:v") + 1] == "copy"
    assert "libx264" not in argv


def test_genpts_is_present(tmp_path: Path):
    """Blocks are concatenated files whose timestamps restart at every join.
    Without +genpts a reader sees only the first block."""
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert "-fflags" in argv
    assert argv[argv.index("-fflags") + 1] == "+genpts"


def test_input_is_paced_at_real_time(tmp_path: Path):
    """Without -re ffmpeg reads the pipe as fast as the feeder fills it."""
    cfg, paths, profile = setup(tmp_path)
    assert "-re" in build_command(cfg, paths, profile, stream_key="KEY")


def test_audio_comes_from_icecast_and_video_from_the_fifo(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
    assert str(paths.video_fifo) in inputs
    assert cfg.audio.icecast.stream_url in inputs
    assert argv[argv.index("-map") + 1] == "0:v"


def test_the_key_is_appended_to_the_ingest_url(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="abcd-efgh")
    assert argv[-1].endswith("/abcd-efgh")
    assert argv[-1].startswith("rtmp://")


def test_the_output_container_is_flv(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert argv[argv.index("-f", argv.index("-c:a")) + 1] == "flv"


def test_reencode_mode_uses_the_real_encoder(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path, TAD_STREAM__VIDEO_MODE="reencode")
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert argv[argv.index("-c:v") + 1] == "libx264"
    assert "copy" not in argv


def test_the_overlay_forces_a_re_encode_and_adds_a_filter(tmp_path: Path):
    cfg, paths, profile = setup(
        tmp_path,
        TAD_STREAM__VIDEO_MODE="reencode",
        TAD_STREAM__REACTIVE_OVERLAY__ENABLED="true",
    )
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert "-filter_complex" in argv
    assert "showcqt" in argv[argv.index("-filter_complex") + 1]
    assert argv[argv.index("-c:v") + 1] != "copy"


def test_reconnect_flags_survive_an_icecast_restart(tmp_path: Path):
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert "-reconnect" in argv
    assert "-reconnect_streamed" in argv


def test_progress_goes_to_stdout_not_stderr(tmp_path: Path):
    """The stderr layout is not a contract; -progress key=value is."""
    cfg, paths, profile = setup(tmp_path)
    argv = build_command(cfg, paths, profile, stream_key="KEY")
    assert argv[argv.index("-progress") + 1] == "pipe:1"


# --- health parsing ------------------------------------------------------


def test_health_parses_ffmpeg_progress():
    from app.services.stream.streamer import FfmpegStreamer

    streamer = FfmpegStreamer.__new__(FfmpegStreamer)
    streamer.health = StreamHealth()
    for key, value in [
        ("frame", "1234"),
        ("fps", "30.1"),
        ("bitrate", "3712.4kbits/s"),
        ("drop_frames", "2"),
        ("dup_frames", "1"),
        ("speed", "1.01x"),
        ("out_time_us", "41000000"),
    ]:
        streamer._apply_progress(key, value)

    assert streamer.health.frames == 1234
    assert streamer.health.fps == pytest.approx(30.1)
    assert streamer.health.bitrate_kbits == pytest.approx(3712.4)
    assert streamer.health.dropped_frames == 2
    assert streamer.health.speed == pytest.approx(1.01)
    assert streamer.health.out_time_s == pytest.approx(41.0)


def test_health_ignores_the_not_available_values_ffmpeg_emits_first():
    from app.services.stream.streamer import FfmpegStreamer

    streamer = FfmpegStreamer.__new__(FfmpegStreamer)
    streamer.health = StreamHealth()
    streamer._apply_progress("fps", "N/A")
    streamer._apply_progress("bitrate", "N/A")
    assert streamer.health.fps == 0.0
    assert streamer.health.updated_at == 0.0


# --- now playing ---------------------------------------------------------


def test_nowplaying_round_trip(tmp_path: Path):
    path = tmp_path / "np.json"
    nowplaying.write(path, nowplaying.NowPlaying(artist="Aurora", title="Drive"))
    got = nowplaying.read(path)
    assert got.artist == "Aurora"
    assert got.display == "Aurora — Drive"


def test_nowplaying_falls_back_to_the_title_alone(tmp_path: Path):
    path = tmp_path / "np.json"
    path.write_text(json.dumps({"title": "Untitled"}), encoding="utf-8")
    assert nowplaying.read(path).display == "Untitled"


def test_a_missing_or_torn_file_is_not_an_error(tmp_path: Path):
    assert nowplaying.read(tmp_path / "nope.json").display == "—"
    torn = tmp_path / "torn.json"
    torn.write_text('{"artist": "half', encoding="utf-8")
    assert nowplaying.read(torn).display == "—"


def test_elapsed_is_never_negative(tmp_path: Path):
    import time

    path = tmp_path / "np.json"
    nowplaying.write(path, nowplaying.NowPlaying(title="x", started_at=time.time() + 500))
    assert nowplaying.read(path).elapsed_s == 0.0


def _manager(tmp_path: Path):
    from app.core.paths import Paths
    from app.core.runtime import Runtime
    from app.core.security import SecretStore
    from app.services.stream.manager import StreamManager

    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    runtime = Runtime(
        config=cfg, paths=paths, secrets=SecretStore.open(paths.secrets_file), service="test"
    )
    return StreamManager(runtime)


def test_shutting_down_keeps_the_intent_to_be_live(tmp_path: Path):
    """Going away is not the same as being told to go off air. Recording
    "stopped" on shutdown is what made resume_on_start do nothing."""
    manager = _manager(tmp_path)
    manager.state.want_live = True
    manager.stop(remember=False)
    assert manager.wanted_live()


def test_being_told_to_stop_is_remembered(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.state.want_live = True
    manager.stop()
    assert not manager.wanted_live()


def test_the_broadcast_remembers_that_it_was_live(tmp_path: Path):
    """StreamManager already wrote want_live to its state file; nothing ever
    read it back, so a restarted container came up off air."""
    from app.services.stream.manager import StreamManager

    _, paths, _ = setup(tmp_path)
    paths.stream_state_file.write_text(json.dumps({"want_live": True}), encoding="utf-8")
    assert StreamManager.wanted_live(_FakeManager(paths))


def test_a_missing_or_broken_state_file_means_not_live(tmp_path: Path):
    from app.services.stream.manager import StreamManager

    _, paths, _ = setup(tmp_path)
    assert not StreamManager.wanted_live(_FakeManager(paths))
    paths.stream_state_file.write_text("{not json", encoding="utf-8")
    assert not StreamManager.wanted_live(_FakeManager(paths))


class _FakeManager:
    """Just enough of StreamManager to exercise the state file reader."""

    def __init__(self, paths):
        self.paths = paths


def test_coming_up_without_a_key_stops_claiming_to_want_live(tmp_path: Path, monkeypatch):
    """Otherwise the state file goes on saying the station wants to be live
    forever, and the next key entered anywhere puts it on air unasked."""
    from fastapi.testclient import TestClient

    from app.core import db as db_module
    from app.core.paths import Paths

    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAD_AUTH__ENABLED", "false")
    monkeypatch.setenv("TAD_CONFIG_FILE", str(tmp_path / "no-such-config.yaml"))
    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()
    paths.stream_state_file.write_text(json.dumps({"want_live": True}), encoding="utf-8")

    from app.main import create_app

    with TestClient(create_app()):
        pass
    db_module.dispose()

    assert not json.loads(paths.stream_state_file.read_text(encoding="utf-8"))["want_live"]


def test_a_title_nobody_can_deliver_is_a_warning(tmp_path: Path):
    """RTMP carries video and audio and nothing else. A title set without a
    connected account is saved, shown back, and then quietly ignored every
    time the station goes live — which is exactly what happened on the first
    real broadcast."""
    from app.core.paths import Paths
    from app.core.runtime import SECRET_YT_CLIENT_ID, SECRET_YT_REFRESH_TOKEN
    from app.core.security import SecretStore
    from app.services.system.preflight import Report, Status, _check_broadcast_title

    cfg = load_config(None, environ={"TAD_APP__DATA_DIR": str(tmp_path / "data")})
    paths = Paths.from_config(cfg)
    paths.ensure()

    report = Report()
    _check_broadcast_title(report, cfg, paths)
    assert report.checks == [], "no title configured is not worth a line"

    cfg.stream.youtube.title_template = "Trance AutoDJ — {datetime}"
    report = Report()
    _check_broadcast_title(report, cfg, paths)
    assert [c.status for c in report.checks] == [Status.WARN]
    assert "no YouTube account is connected" in report.checks[0].detail

    store = SecretStore.open(paths.secrets_file)
    store.set(SECRET_YT_CLIENT_ID, "client-id")
    store.set(SECRET_YT_REFRESH_TOKEN, "refresh-token")
    report = Report()
    _check_broadcast_title(report, cfg, paths)
    assert [c.status for c in report.checks] == [Status.OK]


def test_a_late_monitor_teardown_leaves_a_live_broadcast_alone(tmp_path: Path):
    """The monitor and the broadcast share one feeder, and the teardown
    arrives late by nature: the browser's response generator only unwinds when
    the socket gives way, which can be minutes after going live already took
    the monitor down. Stopping the feeder there killed a real broadcast five
    minutes in — ffmpeg stayed up with nothing to send, raised no error, and
    the watchdog saw a process that was still running."""
    manager = _manager(tmp_path)

    stopped = []
    manager.feeder.stop = lambda *a, **k: stopped.append(True)  # type: ignore[method-assign]

    process = _ExitedProcess()
    manager._monitor = process
    manager.streamer._process = process
    manager.streamer._stop.clear()
    assert manager.streamer.running

    manager._release_monitor(process, [])
    assert stopped == [], "a late monitor teardown took the broadcast off the air"
    assert manager._monitor is None


def test_a_monitor_teardown_off_air_still_stops_the_feeder(tmp_path: Path):
    manager = _manager(tmp_path)
    stopped = []
    manager.feeder.stop = lambda *a, **k: stopped.append(True)  # type: ignore[method-assign]

    process = _ExitedProcess()
    manager._monitor = process
    manager._release_monitor(process, [])
    assert stopped == [True]


def test_a_broadcast_with_nothing_feeding_it_is_restarted(tmp_path: Path, monkeypatch):
    """The stall detector cannot see this one: the Icecast input never ends,
    so ffmpeg goes on reporting progress for a stream that has no pictures."""
    import app.services.stream.manager as manager_module
    from app.services.stream.manager import FEEDER_GRACE_S

    monkeypatch.setattr(manager_module, "WATCHDOG_POLL_S", 0.05)
    manager = _manager(tmp_path)
    manager._want_live.set()
    manager.streamer._process = _ExitedProcess()
    manager.streamer._stop.clear()
    manager.streamer._started_at = time.time() - (FEEDER_GRACE_S + 1)
    assert manager.streamer.running and not manager.feeder.running

    restarts = []
    manager._restart_with_backoff = lambda: restarts.append(True)  # type: ignore[method-assign]
    threading.Timer(1.0, manager._want_live.clear).start()
    manager._watch()
    assert restarts, "a live broadcast with a dead feeder was left alone"


def test_a_broadcast_that_has_only_just_started_is_left_alone(tmp_path: Path, monkeypatch):
    """_launch starts ffmpeg before the feeder on purpose — opening the FIFO
    for writing blocks until a reader exists."""
    import app.services.stream.manager as manager_module

    monkeypatch.setattr(manager_module, "WATCHDOG_POLL_S", 0.05)
    manager = _manager(tmp_path)
    manager._want_live.set()
    manager.streamer._process = _ExitedProcess()
    manager.streamer._stop.clear()
    manager.streamer._started_at = time.time()

    restarts = []
    manager._restart_with_backoff = lambda: restarts.append(True)  # type: ignore[method-assign]
    threading.Timer(0.5, manager._want_live.clear).start()
    manager._watch()
    assert restarts == []


class _ExitedProcess:
    """A Popen-shaped stand-in that is alive until asked to die."""

    returncode = None

    def poll(self):
        return None

    def send_signal(self, _sig):
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = 0


def test_a_relaunch_without_a_key_goes_off_air(tmp_path: Path):
    """The key taken at start used to live on in the manager, so clearing it
    from the dashboard changed nothing: the watchdog kept relaunching ffmpeg
    against an ingest that rejected it in half a second, once a minute, and
    every relaunch pulled the FIFO away from anyone watching the monitor.
    799 restarts on a station nobody could take off air."""
    from app.core.runtime import SECRET_STREAM_KEY

    manager = _manager(tmp_path)
    manager.runtime.secrets.set(SECRET_STREAM_KEY, "a-real-looking-key")
    started, _ = manager.start()
    assert started
    manager.stop()

    manager._want_live.set()
    manager._backoff = 0.0
    manager.runtime.secrets.delete(SECRET_STREAM_KEY)

    launched = []
    manager._launch = lambda: launched.append(True)  # type: ignore[method-assign]
    manager._restart_with_backoff()

    assert launched == [], "relaunched with a key that no longer exists"
    assert not manager._want_live.is_set()
    assert "stream key" in manager.state.last_error


def test_clearing_the_key_takes_the_station_off_air(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core import db as db_module

    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAD_AUTH__ENABLED", "false")
    monkeypatch.setenv("TAD_CONFIG_FILE", str(tmp_path / "no-such-config.yaml"))
    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        assert client.put("/api/stream/key", json={"key": "a-real-looking-key"}).status_code == 200

        stream = app.state.stream
        stream._want_live.set()
        stream.state.want_live = True

        response = client.delete("/api/stream/key")
        assert response.status_code == 200
        assert response.json() == {"configured": False, "stopped": True}
        assert not stream._want_live.is_set()
    db_module.dispose()


def test_a_destination_that_keeps_refusing_says_so(tmp_path: Path):
    """Every one of these runs died before it sent a frame. The dashboard
    showed nothing but an exit code, which reads the same whether YouTube is
    down for a minute or the key is wrong for a fortnight."""
    from app.services.stream.manager import REFUSED_STREAK

    manager = _manager(tmp_path)
    manager._launched_at = time.time()
    for _ in range(REFUSED_STREAK - 1):
        manager._on_ffmpeg_exit(251)
        assert "refused" not in manager.state.last_error

    manager._on_ffmpeg_exit(251)
    assert "refused" in manager.state.last_error
    assert "stream key" in manager.state.last_error


def test_a_run_that_lasted_clears_the_suspicion(tmp_path: Path):
    """A station that drops once an hour is a network, not a bad key."""
    from app.services.stream.manager import HEALTHY_AFTER_S, REFUSED_STREAK

    manager = _manager(tmp_path)
    manager._launched_at = time.time()
    for _ in range(REFUSED_STREAK + 2):
        manager._on_ffmpeg_exit(251)
    assert "refused" in manager.state.last_error

    manager._launched_at = time.time() - (HEALTHY_AFTER_S + 1)
    manager._on_ffmpeg_exit(251)
    assert "refused" not in manager.state.last_error


def test_the_monitor_uses_a_movflag_that_exists(tmp_path: Path):
    """`default_base_is_moof` is not a flag in ffmpeg 8 — the name is
    `default_base_moof` — and the muxer rejects the whole option string, so
    the monitor died in a third of a second with its reason on a stderr
    nobody was reading."""
    from app.services.stream.streamer import build_monitor_command

    cfg, paths, profile = setup(tmp_path)
    argv = build_monitor_command(cfg, paths, profile)
    flags = argv[argv.index("-movflags") + 1]
    assert "default_base_is_moof" not in flags
    assert "default_base_moof" in flags
    assert "empty_moov" in flags, "a live stream has no index to put at the end"


def test_the_monitor_encodes_exactly_what_the_broadcast_does(tmp_path: Path):
    """A preview encoded differently from the thing it previews invites you to
    trust it about something it cannot know."""
    from app.services.stream.streamer import build_monitor_command

    cfg, paths, profile = setup(tmp_path)
    live = build_command(cfg, paths, profile, stream_key="KEY")
    monitor = build_monitor_command(cfg, paths, profile)
    for flag in ("-c:v", "-c:a", "-b:a", "-ar", "-ac"):
        assert live[live.index(flag) + 1] == monitor[monitor.index(flag) + 1], flag
    assert "-progress" not in monitor, "stdout is carrying the video"
    assert monitor[-1] == "pipe:1"


def test_the_monitor_can_be_stopped_from_outside_its_own_request(tmp_path: Path):
    """A browser that closes the tab does not always say so — the kernel
    buffers the writes nobody is reading — so ffmpeg could run on for minutes
    holding the FIFO that going live needs."""

    manager = _manager(tmp_path)
    assert manager.stop_monitor() is False  # nothing running, and no exception


def test_a_monitor_nobody_is_watching_gives_up(tmp_path: Path):
    """Neither a closed socket nor `is_disconnected()` reports a browser that
    has gone: the kernel keeps accepting writes into a buffer. So the viewer
    has to keep saying it is there, and stop being believed when it stops."""
    manager = _manager(tmp_path)
    manager.monitor_heartbeat()
    assert not manager.monitor_abandoned(idle_s=60)
    assert manager.monitor_abandoned(idle_s=-1)


def test_nothing_is_abandoned_before_it_starts(tmp_path: Path):
    manager = _manager(tmp_path)
    assert not manager.monitor_abandoned(idle_s=-1)


def test_a_mount_that_is_not_up_yet_is_waited_for(tmp_path: Path):
    """Icecast answers 404 for a mount with no source. The plain -reconnect
    flags only cover a stream that drops *after* it opened, so a 404 at open
    time killed ffmpeg outright — which is precisely the state in the seconds
    after a restart, while Liquidsoap is still connecting."""
    from app.services.stream.streamer import build_monitor_command

    cfg, paths, profile = setup(tmp_path)
    for argv in (
        build_command(cfg, paths, profile, stream_key="KEY"),
        build_monitor_command(cfg, paths, profile),
    ):
        assert "-reconnect_on_http_error" in argv
        assert "404" in argv[argv.index("-reconnect_on_http_error") + 1]


def test_changing_a_stream_setting_reaches_the_streamer(tmp_path: Path):
    """The manager, the feeder, the pool and the streamer were each built with
    a reference to one Config object. Changing a stream setting from the
    dashboard updated everything except the parts doing the streaming: the
    RTMP target, the audio bitrate and the video mode kept whatever the
    container had started with, with nothing said about it.
    """
    manager = _manager(tmp_path)
    assert manager.cfg.stream.audio.bitrate_k == 192

    changed = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_STREAM__AUDIO__BITRATE_K": "256",
            "TAD_STREAM__YOUTUBE__RTMP_URL": "rtmp://elsewhere/live",
        },
    )
    needs_restart = manager.apply_settings(changed)

    assert needs_restart is False, "nothing is live, so nothing needs restarting"
    for holder in (manager, manager.feeder, manager.feeder.pool, manager.streamer):
        assert holder.cfg.stream.audio.bitrate_k == 256, holder

    argv = build_command(manager.cfg, manager.paths, manager.profile, stream_key="K")
    assert argv[-1] == "rtmp://elsewhere/live/K"
    assert argv[argv.index("-b:a") + 1] == "256k"


# -- naming the broadcast --------------------------------------------------


def test_the_timestamp_is_in_the_stations_timezone():
    """A broadcast started at 03:00 in Athens should say 03:00. The container
    clock is UTC, so taking the naive local time would name it 00:00."""
    from datetime import UTC, datetime

    from app.services.stream.youtube import render_title

    midnight_utc = datetime(2026, 9, 10, 0, 30, tzinfo=UTC)
    assert render_title("Live {datetime}", timezone="Europe/Athens", when=midnight_utc) == (
        "Live 10/09/2026 03:30"
    )
    assert render_title("Live {datetime}", timezone="UTC", when=midnight_utc) == (
        "Live 10/09/2026 00:30"
    )


def test_each_placeholder_is_replaced():
    from datetime import UTC, datetime

    from app.services.stream.youtube import render_title

    when = datetime(2026, 9, 10, 18, 5, tzinfo=UTC)
    assert render_title("{date}", timezone="UTC", when=when) == "10/09/2026"
    assert render_title("{time}", timezone="UTC", when=when) == "18:05"
    assert render_title("A — {datetime}", timezone="UTC", when=when) == "A — 10/09/2026 18:05"


def test_an_empty_template_leaves_the_title_alone():
    """Anyone who has not connected an account must not have their broadcast
    renamed to an empty string."""
    from app.services.stream.youtube import render_title

    assert render_title("", timezone="Europe/Athens") == ""


def test_a_nonsense_timezone_still_produces_a_title():
    """A bad timezone is a typo in a settings box, not a reason to fail to
    name a broadcast."""
    from datetime import UTC, datetime

    from app.services.stream.youtube import render_title

    when = datetime(2026, 9, 10, 18, 5, tzinfo=UTC)
    assert render_title("{time}", timezone="Mars/Olympus", when=when) == "18:05"


def test_renaming_keeps_the_rest_of_the_snippet(monkeypatch):
    """liveBroadcasts.update replaces the whole snippet, so anything left out
    is erased — the scheduled start time included, which YouTube then
    rejects."""
    from app.services.stream import youtube

    sent = {}

    def fake_api(method, path, token, *, params, body=None):
        sent.update({"method": method, "path": path, "body": body})
        return {}

    monkeypatch.setattr(youtube, "_api", fake_api)
    broadcast = {
        "id": "abc",
        "snippet": {"title": "old", "scheduledStartTime": "2026-09-10T00:00:00Z"},
    }
    youtube.rename("tok", broadcast, "new name", "")

    assert sent["method"] == "PUT"
    assert sent["body"]["snippet"]["title"] == "new name"
    assert sent["body"]["snippet"]["scheduledStartTime"] == "2026-09-10T00:00:00Z"


def test_a_title_longer_than_youtube_allows_is_cut(monkeypatch):
    from app.services.stream import youtube

    sent = {}
    monkeypatch.setattr(youtube, "_api", lambda *a, **k: sent.update(k) or {})
    youtube.rename("tok", {"id": "x", "snippet": {}}, "y" * 250, "")
    assert len(sent["body"]["snippet"]["title"]) == 100


def test_waiting_for_approval_is_not_an_error(monkeypatch):
    """`authorization_pending` is the normal answer for as long as the
    operator has not typed the code yet."""
    from app.services.stream import youtube

    monkeypatch.setattr(youtube, "_post", lambda *a, **k: {"error": "authorization_pending"})
    assert youtube.poll_device_flow("id", "secret", "device") is None

    monkeypatch.setattr(youtube, "_post", lambda *a, **k: {"error": "access_denied"})
    with pytest.raises(youtube.YoutubeError):
        youtube.poll_device_flow("id", "secret", "device")
