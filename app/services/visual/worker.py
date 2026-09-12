"""The offline visual generator.

Runs as its own process at ``nice 19`` under a CPU quota, wakes up, asks the
pool whether it needs anything, builds one block if so, then sleeps. It never
runs in parallel with itself and never competes with the broadcast.

Nothing here is on the live path. If this process dies, the stream keeps
playing whatever is already in the pool — which is the entire reason the
architecture separates production from broadcast.
"""

from __future__ import annotations

import random
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlmodel import Session

from app.core.config import Config
from app.core.db import session_scope
from app.core.logging import get_logger
from app.core.paths import Paths
from app.core.runtime import Runtime, build_runtime
from app.models.entities import GenerationJob, JobKind, JobStatus, utcnow
from app.services.system import throttle
from app.services.visual import ai_source, procedural
from app.services.visual.animator import Clip, pick_clip_duration, render_clip
from app.services.visual.blockbuilder import build_block
from app.services.visual.encoder import EncodeProfile
from app.services.visual.palettes import PaletteSet
from app.services.visual.pool import BlockPool

log = get_logger(__name__)

SERVICE_NAME = "generator"

# How long to sleep when there is nothing to do. Long enough that an idle
# machine stays idle; short enough that a drained pool refills promptly.
IDLE_SLEEP_S = 60.0

# Clips are dropped once their block is built. Anything older than this is
# debris from a crashed run.
CLIP_DEBRIS_AGE_S = 6 * 3600


@dataclass
class BlockPlan:
    """Which generators, palettes and durations make up one block."""

    durations: list[float]
    generators: list[str]
    seeds: list[int]
    #: True where the clip should come from AI stills rather than a generator.
    from_ai: list[bool] = field(default_factory=list)

    @property
    def clip_count(self) -> int:
        return len(self.durations)

    @property
    def ai_clips(self) -> int:
        return sum(self.from_ai)


def plan_block(cfg: Config, rng: random.Random, *, ai_available: bool = False) -> BlockPlan:
    """Choose enough clips to fill the target block duration.

    Clips overlap during transitions, so the plan aims a little long; the
    exact length is whatever the crossfades leave.
    """
    enabled = [name for name in cfg.visual.procedural.generators if name in procedural.available()]
    unknown = set(cfg.visual.procedural.generators) - set(procedural.available())
    if unknown:
        log.warning("ignoring unknown generators: %s", ", ".join(sorted(unknown)))
    if not enabled:
        enabled = sorted(procedural.available())
        log.warning("no configured generator exists; using all of them")

    target = cfg.visual.block.duration_s
    average_transition = (
        cfg.visual.block.transition_min_s + cfg.visual.block.transition_max_s
    ) / 2.0

    # The AI share only applies when a backend actually loaded. Asking for
    # 60% AI on a machine with no model must produce a full block of
    # procedural visuals, not a block that is 40% long.
    ai_ratio = cfg.visual.sources.ai_ratio if ai_available else 0.0

    durations: list[float] = []
    generators: list[str] = []
    seeds: list[int] = []
    from_ai: list[bool] = []
    accumulated = 0.0

    while accumulated < target:
        duration = pick_clip_duration(cfg, rng)
        durations.append(duration)
        # Avoid the same generator twice in a row: the crossfade between two
        # clips of the same thing reads as a glitch rather than a transition.
        choices = [g for g in enabled if not generators or g != generators[-1]] or enabled
        generators.append(rng.choice(choices))
        seeds.append(rng.randrange(1, 2**31))
        from_ai.append(rng.random() < ai_ratio)
        accumulated += duration if len(durations) == 1 else duration - average_transition

    return BlockPlan(durations=durations, generators=generators, seeds=seeds, from_ai=from_ai)


def build_one_block(
    runtime: Runtime,
    *,
    profile: EncodeProfile,
    palettes: PaletteSet,
    pool: BlockPool,
    rng: random.Random | None = None,
    job_id: int | None = None,
    should_stop: threading.Event | None = None,
    ai: ai_source.AiSource | None = None,
) -> str | None:
    """Render the clips for one block, join them, and register the result."""
    cfg = runtime.config
    paths = runtime.paths
    rng = rng or random.Random()
    plan = plan_block(cfg, rng, ai_available=bool(ai and ai.available))

    log.info(
        "starting a block",
        extra={
            "clips": plan.clip_count,
            "generators": sorted(set(plan.generators)),
            "ai_clips": plan.ai_clips,
        },
    )

    clips: list[Clip] = []
    stills: list = []
    try:
        for index, name in enumerate(plan.generators):
            if should_stop is not None and should_stop.is_set():
                log.info("stopping before clip %d as asked", index)
                return None
            # An AI slot that produces nothing must still produce a clip.
            factory = None
            if plan.from_ai[index] and ai is not None and ai.available:
                factory, made = _still_factory(ai, cfg, paths, rng)
                stills.extend(made)
                if factory is None:
                    log.warning("no stills for slot %d; using a generator", index)
            if factory is None:
                factory = procedural.get(name)

            clip = render_clip(
                factory,
                cfg=cfg,
                profile=profile,
                palette=palettes.choose(rng),
                duration_s=plan.durations[index],
                seed=plan.seeds[index],
                out_dir=paths.clips,
            )
            clips.append(clip)
            if job_id is not None:
                _update_job(
                    job_id,
                    progress=(index + 1) / (plan.clip_count + 1),
                    step=f"clip {index + 1}/{plan.clip_count}",
                )

        if job_id is not None:
            _update_job(job_id, progress=0.95, step="joining clips")

        built = build_block(
            clips,
            cfg=cfg,
            profile=profile,
            blocks_dir=paths.blocks,
            thumbnails_dir=paths.thumbnails,
            rejected_dir=paths.blocks_rejected,
            source="procedural",
            rng=rng,
        )
    finally:
        # The clips exist only to be joined; keeping them would double the
        # pool's disk use for no benefit. The stills go the same way: their
        # prompts and seeds live on in the block's sidecar.
        for clip in clips:
            clip.path.unlink(missing_ok=True)
        ai_source.cleanup(stills)

    with session_scope() as session:
        pool.register(session, built, source="procedural")
        pool.prune(session)

    return built.id if built.valid else None


def _still_factory(
    ai: ai_source.AiSource,
    cfg: Config,
    paths: Paths,
    rng: random.Random,
):
    """Generate stills and wrap them in an animator factory, or give up."""
    from app.services.visual.animator import render_size

    width, height = render_size(cfg)
    images = ai_source.generate_images(
        ai, cfg=cfg, paths=paths, count=cfg.visual.ai.images_per_clip, rng=rng
    )
    if not images:
        return None, []
    return ai_source.animator_factory(images, width=width, height=height), images


def _update_job(job_id: int, *, progress: float | None = None, step: str | None = None) -> None:
    with session_scope() as session:
        job = session.get(GenerationJob, job_id)
        if job is None:
            return
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        if step is not None:
            job.step = step
        session.add(job)


def _create_job(session: Session, kind: JobKind = JobKind.BLOCK) -> GenerationJob:
    job = GenerationJob(kind=kind, status=JobStatus.RUNNING, started_at=utcnow())
    session.add(job)
    session.flush()
    return job


def sweep_clip_debris(paths: Paths, max_age_s: float = CLIP_DEBRIS_AGE_S) -> int:
    """Delete intermediate clips left behind by an interrupted run."""
    now = time.time()
    removed = 0
    for path in paths.clips.glob("clip_*.mkv"):
        try:
            if now - path.stat().st_mtime > max_age_s:
                path.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("cleared %d abandoned clip(s)", removed)
    return removed


class GeneratorWorker:
    """The long-running loop. One block at a time, forever."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.cfg = runtime.config
        self.profile = EncodeProfile.from_config(runtime.config)
        self.palettes = PaletteSet.load(Path(runtime.config.app.config_dir) / "palettes.json")
        self.pool = BlockPool(runtime.config, runtime.paths, self.profile)
        self.ai = ai_source.build(runtime.config, runtime.paths)
        self.stop_event = threading.Event()
        self.rng = random.Random()

    def request_stop(self, *_args: object) -> None:
        log.info("shutdown requested; will stop after the current block")
        self.stop_event.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)

    def run_once(self) -> str | None:
        """One pass: reconcile, decide, maybe build. Returns a block id."""
        with session_scope() as session:
            self.pool.reconcile(session)
            stats = self.pool.stats(session)

        if stats.at_capacity:
            log.debug("pool is full", extra=stats.as_dict())
            # The pool is full, so this process has nothing else to do with
            # the CPU budget it already has. Measuring a few tempos is exactly
            # the kind of work that should happen now and never during a
            # broadcast.
            self._analyse_some()
            return None

        decision = throttle.evaluate(self.cfg, streaming=self._is_streaming())
        if not decision.allowed:
            log.info("waiting", extra=decision.as_dict())
            return None

        with session_scope() as session:
            job = _create_job(session)
            job_id = job.id

        try:
            block_id = build_one_block(
                self.runtime,
                profile=self.profile,
                palettes=self.palettes,
                pool=self.pool,
                rng=self.rng,
                job_id=job_id,
                should_stop=self.stop_event,
                ai=self.ai,
            )
        except Exception as exc:
            log.exception("block generation failed")
            with session_scope() as session:
                job = session.get(GenerationJob, job_id)
                if job is not None:
                    job.status = JobStatus.FAILED
                    job.error = str(exc)[:2000]
                    job.finished_at = utcnow()
                    session.add(job)
            return None

        with session_scope() as session:
            job = session.get(GenerationJob, job_id)
            if job is not None:
                job.status = JobStatus.DONE if block_id else JobStatus.CANCELLED
                job.progress = 1.0
                job.step = "done"
                job.result_id = block_id or ""
                job.finished_at = utcnow()
                session.add(job)
        return block_id

    def _analyse_some(self, limit: int = 5) -> None:
        """Use idle time to measure tempos. Never fatal."""
        try:
            from app.services.audio import library, playlists

            with session_scope() as session:
                result = library.analyse_pending(session, self.runtime.paths, self.cfg, limit=limit)
                if result.get("analysed"):
                    active = playlists.active(session)
                    if active is not None:
                        playlists.write_m3u(session, active, self.runtime.paths, self.cfg)
        except Exception:
            log.exception("idle tempo analysis failed")

    def _is_streaming(self) -> bool:
        state = self.runtime.paths.stream_state_file
        if not state.is_file():
            return False
        try:
            import json

            return bool(json.loads(state.read_text(encoding="utf-8")).get("live"))
        except (OSError, ValueError):
            return False

    def run(self) -> None:
        self.install_signal_handlers()
        sweep_clip_debris(self.runtime.paths)
        log.info(
            "generator started",
            extra={
                "generators": sorted(procedural.available()),
                "palettes": len(self.palettes),
                "encoder": self.profile.encoder,
                "resolution": self.profile.resolution,
                "ai": self.ai.as_dict(),
            },
        )

        while not self.stop_event.is_set():
            produced = self.run_once()
            if self.stop_event.is_set():
                break
            # Only idle when nothing was produced; a pool below its minimum
            # should refill back to back.
            wait = 1.0 if produced else min(IDLE_SLEEP_S, self.cfg.cpu.throttle_poll_s)
            self.stop_event.wait(wait)

        log.info("generator stopped")


def main() -> None:
    runtime = build_runtime(service=SERVICE_NAME)
    GeneratorWorker(runtime).run()


if __name__ == "__main__":  # pragma: no cover
    main()
