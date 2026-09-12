"""Owns the broadcast: feeder, ffmpeg, and the watchdog that keeps them up.

A 24/7 stream fails in a hundred small ways — the network blips, YouTube drops
the connection, Liquidsoap restarts, a block file goes missing. None of those
may end the broadcast, so everything here is built around restarting rather
than reporting.
"""

from __future__ import annotations

import contextlib
import json
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Config
from app.core.logging import get_logger
from app.core.paths import Paths
from app.core.runtime import (
    SECRET_YT_CLIENT_ID,
    SECRET_YT_CLIENT_SECRET,
    SECRET_YT_REFRESH_TOKEN,
    Runtime,
)
from app.services.stream import youtube
from app.services.stream.feeder import Feeder
from app.services.stream.streamer import FfmpegStreamer, build_monitor_command
from app.services.visual.encoder import EncodeProfile

log = get_logger(__name__)

# The watchdog checks this often.
WATCHDOG_POLL_S = 5.0

# Messages ffmpeg and the feeder produce *because* we stopped them. They are
# the sound of a clean shutdown, not a fault, and showing them in red on the
# dashboard after every STOP teaches the operator to ignore the error line.
BENIGN_ON_STOP = (
    "ffmpeg closed the pipe",
    "Failed to update header with correct duration",
    "Failed to update header with correct filesize",
    "Exiting normally, received signal 2",
)


def _is_benign(message: str) -> bool:
    return any(phrase in message for phrase in BENIGN_ON_STOP)


# How long the monitor's viewer may go quiet before we stop believing it.
# The page pings every few seconds, so this is several missed pings.
MONITOR_IDLE_S = 20.0

# A run this long counts as healthy and resets the backoff.
HEALTHY_AFTER_S = 60.0

# Relaunches in a row that died before HEALTHY_AFTER_S. A handful of these
# is not the network blipping: it is the ingest refusing us, which is worth
# saying out loud instead of hiding behind an exit code that repeats for
# hours. The broadcast keeps retrying either way — a 24/7 station does not
# give up because YouTube was unhappy at four in the morning.
REFUSED_STREAK = 3

# _launch starts ffmpeg before the feeder — opening the FIFO for writing
# blocks until a reader exists — so a brand-new broadcast is briefly live
# with no feeder, and that is not a fault.
FEEDER_GRACE_S = 10.0


@dataclass
class StreamState:
    """Written to disk so the generator can see whether we are on air, and so
    a restarted process can find out what it was doing before."""

    live: bool = False
    since: float = 0.0
    restarts: int = 0
    last_error: str = ""
    # Intent, not status. `live` says whether ffmpeg is running right now;
    # this says whether anybody wants it to be, which is what has to survive
    # a crash, an upgrade or a reboot.
    want_live: bool = False
    # What we last managed to call the broadcast on YouTube, so the dashboard
    # can show whether the naming actually worked.
    broadcast_title: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "live": self.live,
            "since": self.since,
            "restarts": self.restarts,
            "last_error": self.last_error,
            "want_live": self.want_live,
            "broadcast_title": self.broadcast_title,
        }


class StreamManager:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.cfg: Config = runtime.config
        self.paths: Paths = runtime.paths
        self.profile = EncodeProfile.from_config(runtime.config)
        self.feeder = Feeder(self.cfg, self.paths, self.profile)
        self.streamer = FfmpegStreamer(self.cfg, self.paths, self.profile)
        self.state = StreamState()

        self._lock = threading.Lock()
        self._want_live = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._backoff = self.cfg.stream.watchdog.backoff_min_s
        self._stream_key = ""
        self._launched_at = 0.0
        self._short_runs = 0
        self._monitor: subprocess.Popen[bytes] | None = None
        self._monitor_seen = 0.0

    def apply_settings(self, config: Config) -> bool:
        """Take up changed settings. Says whether a live broadcast must be
        restarted for them to reach the wire.

        The manager, the feeder and the streamer were each built with a
        reference to one Config object, so changing a stream setting from the
        dashboard updated everything except the parts doing the streaming: the
        RTMP target, the audio bitrate, the video mode and the watchdog all
        kept the values the container started with, silently, until it was
        restarted.
        """
        self.cfg = config
        self.profile = EncodeProfile.from_config(config)
        self.feeder.cfg = config
        # The feeder reaches the pool for the next block, and the pool decides
        # what counts as playable from the profile — so it needs both too.
        self.feeder.pool.cfg = config
        self.feeder.pool.profile = self.profile
        self.streamer.cfg = config
        self.streamer.profile = self.profile
        # ffmpeg is given its command line once, at start. A running broadcast
        # keeps the old one until it is restarted, and saying so is better
        # than silently dropping the stream to apply a bitrate change.
        return self.streamer.running

    # -- state file --------------------------------------------------------

    def _write_state(self) -> None:
        try:
            self.paths.state.mkdir(parents=True, exist_ok=True)
            temporary = self.paths.stream_state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state.as_dict()), encoding="utf-8")
            temporary.replace(self.paths.stream_state_file)
        except OSError as exc:
            log.warning("could not write the stream state file: %s", exc)

    def wanted_live(self) -> bool:
        """Was the broadcast wanted when this data directory was last used?

        Nothing used to persist this at all — the state file recorded whether
        ffmpeg happened to be running, which is worthless after the process
        that ran it has gone — so a container that restarted itself came up
        silent and off air while reporting that everything was fine.
        """
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(self.paths.stream_state_file.read_text(encoding="utf-8"))
            return bool(data.get("want_live"))
        return False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        """Go live. Returns (started, reason)."""
        key = self.runtime.stream_key()
        if not key:
            return False, "no YouTube stream key is configured"

        # The monitor holds the same FIFO. A tab left open somewhere must not
        # be the reason a broadcast cannot start.
        if self.stop_monitor():
            for _ in range(50):
                if self._monitor is None:
                    break
                time.sleep(0.1)

        with self._lock:
            if self.streamer.running:
                return True, "already live"

            self._stream_key = key
            self._short_runs = 0
            self._want_live.set()
            self.state.want_live = True
            self._backoff = self.cfg.stream.watchdog.backoff_min_s
            self._launch()

            if self.cfg.stream.watchdog.enabled and (
                self._watchdog is None or not self._watchdog.is_alive()
            ):
                self._watchdog = threading.Thread(
                    target=self._watch, name="stream-watchdog", daemon=True
                )
                self._watchdog.start()

        return True, "started"

    def _launch(self) -> None:
        # ffmpeg first: opening the FIFO for writing blocks until a reader
        # exists, so the feeder would otherwise hang on a pipe nobody reads.
        self.feeder.ensure_fifo()
        self.streamer.start(self._stream_key, on_exit=self._on_ffmpeg_exit)
        self.feeder.start()

        self._launched_at = time.time()
        self.state.live = True
        self.state.since = self._launched_at
        self.state.last_error = ""
        self._write_state()
        log.info("broadcast started")

        # Name it, in the background. YouTube has not created the broadcast
        # yet — that takes a little after ffmpeg connects — and none of this
        # may hold up going live or fail it.
        threading.Thread(target=self._name_broadcast, name="youtube-title", daemon=True).start()

    def _name_broadcast(self) -> None:
        """Give this broadcast the operator's title, with the time it started.

        Every failure here is logged and swallowed: a quota error or an
        expired token must cost a title, never the broadcast.
        """
        template = self.cfg.stream.youtube.title_template
        if not template:
            return

        secrets = self.runtime.secrets
        client_id = secrets.get(SECRET_YT_CLIENT_ID)
        client_secret = secrets.get(SECRET_YT_CLIENT_SECRET)
        refresh = secrets.get(SECRET_YT_REFRESH_TOKEN)
        if not (client_id and client_secret and refresh):
            log.info("a title is configured but no YouTube account is connected")
            return

        # The stamp is the moment we went on air, not the moment the API
        # answered: a slow reply must not put the wrong time in the name.
        started = datetime.now(tz=UTC)
        title = youtube.render_title(template, timezone=self.cfg.app.timezone, when=started)
        description = youtube.render_title(
            self.cfg.stream.youtube.description_template,
            timezone=self.cfg.app.timezone,
            when=started,
        )

        try:
            token = youtube.access_token(client_id, client_secret, refresh)
            broadcast = youtube.wait_for_broadcast(token)
            if broadcast is None:
                log.warning("no live broadcast appeared; the title was left alone")
                return
            youtube.rename(token, broadcast, title, description)
            self.state.broadcast_title = title
            self._write_state()
            log.info("broadcast named", extra={"title": title, "broadcast": broadcast.get("id")})
        except youtube.YoutubeError as exc:
            log.warning("could not set the broadcast title: %s", exc)
        except Exception as exc:
            log.warning("unexpected failure naming the broadcast: %s", exc)

    def _on_ffmpeg_exit(self, code: int) -> None:
        ran_s = time.time() - self._launched_at if self._launched_at else 0.0
        self._short_runs = self._short_runs + 1 if ran_s < HEALTHY_AFTER_S else 0
        message = f"ffmpeg exited with code {code} after {ran_s:.0f}s"
        if self._short_runs >= REFUSED_STREAK:
            # Every one of these runs died before it could send a frame.
            # The dashboard used to show nothing but the exit code, which
            # is the same whether YouTube is down or the key is wrong.
            message = (
                f"{message} \u2014 the ingest has refused the connection "
                f"{self._short_runs} times in a row; check the stream key "
                "and that the channel is ready to receive a live stream"
            )
        self.state.last_error = message
        self.state.live = False
        self._write_state()

    def stop(self, *, remember: bool = True) -> None:
        """Go off air.

        `remember=False` is the application shutting down rather than being
        told to stop: the processes go away, but the intent to be live has to
        outlive them or nothing would ever come back by itself. The event is
        cleared either way, otherwise the watchdog would relaunch ffmpeg while
        we are trying to exit.
        """
        with self._lock:
            self._want_live.clear()
            if remember:
                self.state.want_live = False
            self.streamer.stop()
            self.feeder.stop()
            self.state.live = False
            # Whatever they said on the way out was about being stopped.
            if _is_benign(self.state.last_error):
                self.state.last_error = ""
            if _is_benign(self.feeder.stats.last_error):
                self.feeder.stats.last_error = ""
            self._write_state()
            log.info("broadcast stopped", extra={"remembered": remember})

    def restart(self) -> None:
        self.stop()
        self.start()

    # -- monitor -----------------------------------------------------------

    def monitor_available(self) -> tuple[bool, str]:
        """Can a monitor run right now, and if not, why not."""
        if self.streamer.running:
            return False, "the broadcast is live — the monitor is for before that"
        if self._monitor is not None and self._monitor.poll() is None:
            return False, "the monitor is already running"
        if not self.feeder.has_blocks():
            return False, "there are no video blocks to play yet"
        return True, "ready"

    def monitor_heartbeat(self) -> None:
        """The page says it is still watching.

        Neither a closed socket nor `is_disconnected()` is reliable here: the
        kernel accepts writes into a buffer long after the browser has gone,
        so the disconnect never surfaces. A viewer that has to keep saying it
        is there cannot leave silently.
        """
        self._monitor_seen = time.monotonic()

    def monitor_abandoned(self, *, idle_s: float) -> bool:
        return bool(self._monitor_seen) and (time.monotonic() - self._monitor_seen) > idle_s

    def stop_monitor(self) -> bool:
        """Kill a running monitor from outside its own request.

        Needed because a browser that goes away does not always tell us: the
        kernel buffers the writes nobody is reading, so the disconnect can go
        unnoticed for minutes while ffmpeg keeps running and keeps hold of the
        FIFO the broadcast needs. Going live calls this, and so does the
        dashboard's stop button.
        """
        process = self._monitor
        if process is None or process.poll() is not None:
            return False
        log.info("stopping the monitor")
        with contextlib.suppress(Exception):
            process.send_signal(signal.SIGINT)
        return True

    @contextlib.contextmanager
    def monitor(self) -> Iterator[subprocess.Popen[bytes]]:
        """Run the broadcast for a viewer instead of for YouTube.

        The same feeder, the same ffmpeg, the same codecs — only the container
        and the destination differ. It holds the same FIFO the broadcast uses,
        which is why the two cannot run at once, and why that is the right
        behaviour rather than a limitation: this is what you look at *before*
        going live.
        """
        with self._lock:
            ok, reason = self.monitor_available()
            if not ok:
                raise RuntimeError(reason)
            self.feeder.ensure_fifo()
            argv = build_monitor_command(self.cfg, self.paths, self.profile)
            log.info("monitor starting", extra={"command": " ".join(argv)})
            self._monitor = subprocess.Popen(  # noqa: S603 - argv is a list
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process = self._monitor
            self._monitor_seen = time.monotonic()
            # Without this ffmpeg's reason for giving up is thrown away, and a
            # monitor that dies in a third of a second looks like a browser
            # problem. It cost an hour to learn that "default_base_is_moof" is
            # not a movflag in ffmpeg 8; the name is "default_base_moof".
            errors: list[str] = []
            threading.Thread(
                target=self._drain_monitor_errors,
                args=(process, errors),
                name="monitor-stderr",
                daemon=True,
            ).start()
            self.feeder.start()
            # The response generator cannot police this itself: when the
            # viewer disappears, uvicorn blocks writing into a socket buffer
            # that will not drain, so the generator is suspended inside its
            # `yield` and never runs another check. Something outside it has
            # to notice.
            threading.Thread(
                target=self._reap_abandoned_monitor,
                args=(process,),
                name="monitor-reaper",
                daemon=True,
            ).start()
        try:
            yield process
        finally:
            self._release_monitor(process, errors)

    def _reap_abandoned_monitor(self, process: subprocess.Popen[bytes]) -> None:
        while process.poll() is None:
            if self.monitor_abandoned(idle_s=MONITOR_IDLE_S):
                log.info("nobody is watching the monitor any more; stopping it")
                self._release_monitor(process, [])
                return
            time.sleep(2.0)

    def _release_monitor(self, process: subprocess.Popen[bytes], errors: list[str]) -> None:
        """Tear down one monitor, at most once, whoever gets here first.

        Both the request and the reaper can arrive, and the request can arrive
        very late — long after another monitor has started. Acting on identity
        rather than on "is a monitor running" is what stops a stale teardown
        from killing its successor.
        """
        with self._lock:
            if self._monitor is not process:
                return
            self._monitor = None
            self._monitor_seen = 0.0
        with contextlib.suppress(Exception):
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
        with contextlib.suppress(Exception):
            if process.poll() is None:
                process.kill()

        # The feeder is shared with the broadcast, and this teardown arrives
        # late by nature: the browser's response generator only unwinds when
        # the socket finally gives way, which can be minutes after going live
        # already took the monitor down. Stopping the feeder here killed a
        # real broadcast five minutes in — ffmpeg stayed up with nothing left
        # to send, so no error was ever raised and the watchdog saw a process
        # that was still running. The station simply went quiet.
        if self.streamer.running:
            log.info("the broadcast has the feeder now; leaving it alone")
        else:
            self.feeder.stop()
        log.info(
            "monitor stopped",
            extra={"exit_code": process.returncode, "stderr": " | ".join(errors[-5:])},
        )

    @staticmethod
    def _drain_monitor_errors(process: subprocess.Popen[bytes], sink: list[str]) -> None:
        if process.stderr is None:
            return
        for raw in process.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                sink.append(line)
                log.warning("monitor: %s", line)

    # -- watchdog ----------------------------------------------------------

    def _watch(self) -> None:
        """Restart the broadcast whenever it stops without being asked to."""
        watchdog = self.cfg.stream.watchdog
        while self._want_live.is_set():
            time.sleep(WATCHDOG_POLL_S)
            if not self._want_live.is_set():
                break

            if self.streamer.running:
                if self.streamer.uptime_s >= HEALTHY_AFTER_S:
                    self._backoff = watchdog.backoff_min_s
                    self._short_runs = 0
                # ffmpeg alive with a dead feeder is the one failure the
                # stall detector cannot see: the audio input never ends, so
                # progress keeps being reported for a broadcast that has no
                # pictures left to send. The grace period is there because
                # _launch starts ffmpeg first, on purpose.
                if self.streamer.uptime_s > FEEDER_GRACE_S and not self.feeder.running:
                    log.error("the broadcast is live but nothing is feeding it; restarting")
                    self._restart_with_backoff()
                    continue
                if self._stalled():
                    log.error(
                        "no progress from ffmpeg for %.0fs; restarting",
                        self.streamer.health.stale_s,
                    )
                    self._restart_with_backoff()
                continue

            log.warning(
                "the broadcast is down, restarting in %.0fs",
                self._backoff,
                extra={"restarts": self.state.restarts},
            )
            self._restart_with_backoff()

    def _stalled(self) -> bool:
        """ffmpeg alive but silent for too long is as dead as an exit."""
        health = self.streamer.health
        if not health.updated_at:
            # Nothing has been reported yet; give it a full timeout to start.
            return self.streamer.uptime_s > self.cfg.stream.watchdog.stall_timeout_s * 2
        return health.stale_s > self.cfg.stream.watchdog.stall_timeout_s

    def _restart_with_backoff(self) -> None:
        watchdog = self.cfg.stream.watchdog
        time.sleep(self._backoff)
        if not self._want_live.is_set():
            return

        # Read the key again rather than trusting the copy taken when we
        # first went live. A key cleared from the dashboard used to live on
        # in this attribute, so the watchdog kept relaunching ffmpeg at an
        # ingest that rejected it in half a second — hundreds of times, for
        # as long as the container ran.
        key = self.runtime.stream_key()
        if not key:
            log.error("there is no stream key any more; going off air")
            self.stop()
            self.state.last_error = "the stream key was removed"
            self._write_state()
            return
        self._stream_key = key

        # The monitor holds the same FIFO, so relaunching takes the pipe
        # out from under whoever is watching. Doing it deliberately means
        # they are told the broadcast took over; doing it by accident,
        # once a minute, looks like the station cutting out.
        if self.stop_monitor():
            for _ in range(50):
                if self._monitor is None:
                    break
                time.sleep(0.1)

        with self._lock:
            self.streamer.stop()
            self.feeder.stop()
            self.streamer.note_restart()
            self.state.restarts += 1
            self._launch()
        self._backoff = min(self._backoff * 2, watchdog.backoff_max_s)

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, object]:
        streamer = self.streamer.status().as_dict()
        feeder = self.feeder.stats.as_dict()
        # The counters are the last run's, and saying so is the whole point:
        # a stopped station used to report 30 fps and a block "playing" for
        # longer and longer, because the elapsed time was measured from a
        # start that had already ended.
        feeder["running"] = self.feeder.running
        if not self.feeder.running:
            feeder["current_block_elapsed_s"] = 0.0
        return {
            **streamer,
            "want_live": self._want_live.is_set(),
            "has_stream_key": self.runtime.has_stream_key(),
            "video_mode": self.cfg.stream.video_mode,
            "encoder": self.profile.encoder,
            "feeder": feeder,
            "restarts": self.state.restarts,
        }

    def logs(self, count: int = 100) -> list[str]:
        return self.streamer.tail(count)


__all__ = ["StreamManager", "StreamState"]
