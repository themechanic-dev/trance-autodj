"""The block pool: what exists, what is playable, and what gets deleted.

The filesystem is the source of truth. Database rows describe files; they are
never the only record of one. That way an unmounted disk or a wiped database
costs a rescan, not the library.

Three jobs:

* **reconcile** — bring the database in line with the directory
* **validate** — refuse blocks whose encode profile no longer matches, because
  the streamer copies video and cannot reconcile a mismatch at a join
* **prune** — stay inside the block count and disk quota, deleting the most
  worn-out material first
"""

from __future__ import annotations

import contextlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from app.core.config import Config
from app.core.logging import get_logger
from app.core.paths import Paths
from app.models.entities import Block, BlockSource, BlockStatus, utcnow
from app.services.visual.blockbuilder import BuiltBlock
from app.services.visual.encoder import EncodeProfile, duration_of

log = get_logger(__name__)


@dataclass(frozen=True)
class PoolStats:
    total: int
    ready: int
    rejected: int
    missing: int
    pinned: int
    duration_s: float
    size_bytes: int
    disk_free_gb: float
    min_blocks: int
    max_blocks: int
    max_disk_gb: float

    @property
    def hours_available(self) -> float:
        return self.duration_s / 3600.0

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def below_minimum(self) -> bool:
        return self.ready < self.min_blocks

    @property
    def at_capacity(self) -> bool:
        return self.ready >= self.max_blocks or self.size_gb >= self.max_disk_gb

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "ready": self.ready,
            "rejected": self.rejected,
            "missing": self.missing,
            "pinned": self.pinned,
            "duration_s": round(self.duration_s, 1),
            "hours_available": round(self.hours_available, 2),
            "size_gb": round(self.size_gb, 2),
            "disk_free_gb": round(self.disk_free_gb, 1),
            "min_blocks": self.min_blocks,
            "max_blocks": self.max_blocks,
            "max_disk_gb": self.max_disk_gb,
            "below_minimum": self.below_minimum,
            "at_capacity": self.at_capacity,
        }


class BlockPool:
    def __init__(self, cfg: Config, paths: Paths, profile: EncodeProfile) -> None:
        self.cfg = cfg
        self.paths = paths
        self.profile = profile

    # -- registration ------------------------------------------------------

    def register(self, session: Session, built: BuiltBlock, *, source: str) -> Block:
        """Record a freshly built block."""
        status = BlockStatus.READY if built.valid else BlockStatus.REJECTED
        try:
            relpath = str(built.path.relative_to(self.paths.root))
        except ValueError:
            relpath = str(built.path)

        thumbnail = ""
        if built.thumbnail is not None:
            try:
                thumbnail = str(built.thumbnail.relative_to(self.paths.root))
            except ValueError:
                thumbnail = str(built.thumbnail)

        block = Block(
            id=built.id,
            relpath=relpath,
            duration_s=built.duration_s,
            size_bytes=built.size_bytes,
            source=(
                BlockSource(source)
                if source in BlockSource._value2member_map_
                else BlockSource.PROCEDURAL
            ),
            status=status,
            reject_reason="; ".join(built.problems),
            profile_fingerprint=built.profile_fingerprint,
            thumbnail_relpath=thumbnail,
            metadata_json=json.dumps(
                {
                    "transitions": built.transitions,
                    "clips": [c.as_metadata() for c in built.clips],
                    "build_seconds": round(built.build_seconds, 1),
                },
                ensure_ascii=False,
            ),
        )
        session.merge(block)
        return block

    # -- reconciliation ----------------------------------------------------

    def reconcile(self, session: Session) -> dict[str, int]:
        """Make the database agree with the blocks directory.

        Counts what changed so the caller can log one line instead of one per
        block; on a 60-block pool that is the difference between a readable
        log and a wall of text every minute.
        """
        found = {"added": 0, "missing": 0, "restored": 0, "stale_profile": 0}

        on_disk = {p.stem.removeprefix("block_"): p for p in self.paths.blocks.glob("block_*.ts")}
        rows = {b.id: b for b in session.exec(select(Block)).all()}

        for block_id, path in on_disk.items():
            row = rows.get(block_id)
            if row is None:
                session.add(self._adopt(path, block_id))
                found["added"] += 1
                continue
            if row.status is BlockStatus.MISSING:
                row.status = BlockStatus.READY
                session.add(row)
                found["restored"] += 1

        for block_id, row in rows.items():
            if block_id in on_disk or row.status is BlockStatus.REJECTED:
                continue
            if row.status is not BlockStatus.MISSING:
                row.status = BlockStatus.MISSING
                session.add(row)
                found["missing"] += 1

        # A block encoded under different settings cannot be copied into the
        # same stream as the others; retire it rather than let the streamer
        # discover the mismatch live.
        for row in session.exec(select(Block).where(Block.status == BlockStatus.READY)).all():
            if row.profile_fingerprint and row.profile_fingerprint != self.profile.fingerprint:
                row.status = BlockStatus.REJECTED
                row.reject_reason = "encoded under a different video profile"
                session.add(row)
                found["stale_profile"] += 1

        if any(found.values()):
            log.info("pool reconciled", extra=found)
        return found

    def _adopt(self, path: Path, block_id: str) -> Block:
        """Build a row for a block file found without one, using its sidecar."""
        sidecar = path.with_suffix(".json")
        data: dict = {}
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("unreadable sidecar for %s", path.name)

        duration = float(data.get("duration_s") or 0.0) or duration_of(path, self.cfg.tools.ffprobe)
        fingerprint = str((data.get("profile") or {}).get("fingerprint", ""))
        source = str(data.get("source", "procedural"))
        thumbnail = self.paths.thumbnails / f"block_{block_id}.jpg"

        return Block(
            id=block_id,
            relpath=str(path.relative_to(self.paths.root)),
            duration_s=duration,
            size_bytes=path.stat().st_size,
            source=(
                BlockSource(source)
                if source in BlockSource._value2member_map_
                else BlockSource.PROCEDURAL
            ),
            status=(
                BlockStatus.READY
                if not fingerprint or fingerprint == self.profile.fingerprint
                else BlockStatus.REJECTED
            ),
            reject_reason=(
                ""
                if not fingerprint or fingerprint == self.profile.fingerprint
                else "encoded under a different video profile"
            ),
            profile_fingerprint=fingerprint,
            thumbnail_relpath=(
                str(thumbnail.relative_to(self.paths.root)) if thumbnail.is_file() else ""
            ),
            metadata_json=json.dumps(
                {"transitions": data.get("transitions", []), "clips": data.get("clips", [])},
                ensure_ascii=False,
            ),
        )

    # -- statistics --------------------------------------------------------

    def stats(self, session: Session) -> PoolStats:
        import shutil

        blocks = session.exec(select(Block)).all()
        ready = [b for b in blocks if b.status is BlockStatus.READY]
        usage = shutil.disk_usage(self.paths.root)
        return PoolStats(
            total=len(blocks),
            ready=len(ready),
            rejected=sum(1 for b in blocks if b.status is BlockStatus.REJECTED),
            missing=sum(1 for b in blocks if b.status is BlockStatus.MISSING),
            pinned=sum(1 for b in ready if b.pinned),
            duration_s=sum(b.duration_s for b in ready),
            size_bytes=sum(b.size_bytes for b in ready),
            disk_free_gb=usage.free / 1024**3,
            min_blocks=self.cfg.visual.pool.min_blocks,
            max_blocks=self.cfg.visual.pool.max_blocks,
            max_disk_gb=self.cfg.visual.pool.max_disk_gb,
        )

    # -- pruning -----------------------------------------------------------

    def deletion_order(self, blocks: list[Block]) -> list[Block]:
        """Worst-first: most replayed, longest since played, oldest.

        Never-played blocks sort last. They are the freshest material in the
        pool, and deleting them because they have no play history — which a
        naive "least recently used" would do — would throw away exactly what
        was just made.
        """
        never_played = datetime.max.replace(tzinfo=UTC)

        def key(block: Block) -> tuple:
            last = block.last_played_at
            if last is None:
                last = never_played
            elif last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            return (-block.play_count, last, block.created_at)

        return sorted(blocks, key=key)

    def prune(self, session: Session) -> list[str]:
        """Delete blocks until the pool fits its limits. Returns their ids."""
        pool_cfg = self.cfg.visual.pool
        removed: list[str] = []

        # Rejected and missing rows go first; they cost space and play nothing.
        for block in session.exec(select(Block).where(Block.status == BlockStatus.MISSING)).all():
            session.delete(block)
            removed.append(block.id)

        if not pool_cfg.keep_rejected:
            for block in session.exec(
                select(Block).where(Block.status == BlockStatus.REJECTED)
            ).all():
                self._delete_files(block)
                session.delete(block)
                removed.append(block.id)

        candidates = [
            b
            for b in session.exec(select(Block).where(Block.status == BlockStatus.READY)).all()
            if not b.pinned
        ]
        ready_count = len(candidates) + sum(
            1
            for b in session.exec(select(Block).where(Block.status == BlockStatus.READY)).all()
            if b.pinned
        )
        total_bytes = sum(
            b.size_bytes
            for b in session.exec(select(Block).where(Block.status == BlockStatus.READY)).all()
        )
        max_bytes = int(pool_cfg.max_disk_gb * 1024**3)

        for block in self.deletion_order(candidates):
            if ready_count <= pool_cfg.max_blocks and total_bytes <= max_bytes:
                break
            self._delete_files(block)
            session.delete(block)
            removed.append(block.id)
            ready_count -= 1
            total_bytes -= block.size_bytes

        if removed:
            log.info("pool pruned", extra={"removed": len(removed)})
        return removed

    def _delete_files(self, block: Block) -> None:
        path = self.paths.root / block.relpath
        for target in (path, path.with_suffix(".json")):
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete %s: %s", target, exc)
        if block.thumbnail_relpath:
            with contextlib.suppress(OSError):
                (self.paths.root / block.thumbnail_relpath).unlink(missing_ok=True)

    # -- playback selection ------------------------------------------------

    def pick(
        self, session: Session, count: int = 1, *, rng: random.Random | None = None
    ) -> list[Block]:
        """Weighted shuffle over ready blocks, preferring the least played.

        The streamer uses this so a fresh block is heard from quickly instead
        of waiting out a full cycle of the pool.
        """
        rng = rng or random.Random()
        ready = session.exec(select(Block).where(Block.status == BlockStatus.READY)).all()
        if not ready:
            return []

        strength = self.cfg.stream.play_count_weight
        weights = [1.0 / (1.0 + strength * b.play_count) for b in ready]

        chosen: list[Block] = []
        pool = list(ready)
        pool_weights = list(weights)
        for _ in range(min(count, len(pool))):
            index = rng.choices(range(len(pool)), weights=pool_weights, k=1)[0]
            chosen.append(pool.pop(index))
            pool_weights.pop(index)
        return chosen

    def mark_played(self, session: Session, block_id: str) -> None:
        block = session.get(Block, block_id)
        if block is None:
            return
        block.play_count += 1
        block.last_played_at = utcnow()
        session.add(block)

    def path_of(self, block: Block) -> Path:
        return self.paths.root / block.relpath


__all__ = ["BlockPool", "PoolStats"]
