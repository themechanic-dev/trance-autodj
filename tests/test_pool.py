"""Pool bookkeeping: reconciliation, deletion order, quotas, selection."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.core.config import Config, load_config
from app.core.db import create_tables, dispose, init_engine, session_scope
from app.core.paths import Paths
from app.models.entities import Block, BlockStatus
from app.services.visual.encoder import EncodeProfile
from app.services.visual.pool import BlockPool

FINGERPRINT = Config().video.profile_fingerprint()


@pytest.fixture
def pool_env(tmp_path: Path):
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_VISUAL__POOL__MAX_BLOCKS": "3",
            "TAD_VISUAL__POOL__MIN_BLOCKS": "2",
        },
    )
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    profile = EncodeProfile._build(cfg, "libx264")
    yield cfg, paths, BlockPool(cfg, paths, profile)
    dispose()


def add_block(
    paths: Paths,
    block_id: str,
    *,
    plays: int = 0,
    last_played: datetime | None = None,
    pinned: bool = False,
    size: int = 1024,
    fingerprint: str = FINGERPRINT,
    write_file: bool = True,
) -> Block:
    path = paths.blocks / f"block_{block_id}.ts"
    if write_file:
        path.write_bytes(b"0" * size)
    return Block(
        id=block_id,
        relpath=str(path.relative_to(paths.root)),
        duration_s=600.0,
        size_bytes=size,
        status=BlockStatus.READY,
        profile_fingerprint=fingerprint,
        play_count=plays,
        last_played_at=last_played,
        pinned=pinned,
    )


def test_stats_count_only_ready_blocks(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "a"))
        rejected = add_block(paths, "b")
        rejected.status = BlockStatus.REJECTED
        session.add(rejected)

    with session_scope() as session:
        stats = pool.stats(session)
    assert stats.total == 2
    assert stats.ready == 1
    assert stats.rejected == 1
    assert stats.hours_available == pytest.approx(600 / 3600)


def test_reconcile_adopts_files_that_have_no_row(pool_env):
    _, paths, pool = pool_env
    (paths.blocks / "block_orphan.ts").write_bytes(b"0" * 512)
    (paths.blocks / "block_orphan.json").write_text(
        json.dumps(
            {
                "duration_s": 42.0,
                "source": "procedural",
                "profile": {"fingerprint": FINGERPRINT},
            }
        ),
        encoding="utf-8",
    )
    with session_scope() as session:
        result = pool.reconcile(session)
    assert result["added"] == 1
    with session_scope() as session:
        block = session.get(Block, "orphan")
        assert block is not None
        assert block.duration_s == 42.0
        assert block.status is BlockStatus.READY


def test_reconcile_marks_a_vanished_file_missing_but_keeps_the_row(pool_env):
    """An unmounted disk must cost a rescan, not the library."""
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "gone"))
    (paths.blocks / "block_gone.ts").unlink()

    with session_scope() as session:
        result = pool.reconcile(session)
    assert result["missing"] == 1
    with session_scope() as session:
        assert session.get(Block, "gone").status is BlockStatus.MISSING


def test_reconcile_restores_a_file_that_came_back(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        block = add_block(paths, "back")
        block.status = BlockStatus.MISSING
        session.add(block)
    with session_scope() as session:
        result = pool.reconcile(session)
    assert result["restored"] == 1


def test_a_block_from_another_profile_is_retired(pool_env):
    """It cannot be copied into the same stream as the others."""
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "old", fingerprint="something-else"))
    with session_scope() as session:
        result = pool.reconcile(session)
    assert result["stale_profile"] == 1
    with session_scope() as session:
        block = session.get(Block, "old")
        assert block.status is BlockStatus.REJECTED
        assert "different video profile" in block.reject_reason


def test_deletion_order_sheds_the_most_worn_first(pool_env):
    _, paths, pool = pool_env
    now = datetime.now(UTC)
    blocks = [
        add_block(paths, "fresh", plays=0, last_played=None, write_file=False),
        add_block(paths, "played_once", plays=1, last_played=now, write_file=False),
        add_block(paths, "worn", plays=9, last_played=now - timedelta(days=2), write_file=False),
        add_block(paths, "worn_recent", plays=9, last_played=now, write_file=False),
    ]
    order = [b.id for b in pool.deletion_order(blocks)]
    assert order[0] == "worn"  # most plays, longest ago
    assert order[1] == "worn_recent"  # most plays, but played recently
    assert order[-1] == "fresh"  # never played: newest material, keep


def test_never_played_blocks_are_not_deleted_first(pool_env):
    """A naive 'least recently used' would throw away what was just built."""
    _, paths, pool = pool_env
    now = datetime.now(UTC)
    blocks = [
        add_block(paths, "new1", plays=0, last_played=None, write_file=False),
        add_block(paths, "old1", plays=3, last_played=now - timedelta(days=5), write_file=False),
    ]
    assert pool.deletion_order(blocks)[0].id == "old1"


def test_prune_respects_max_blocks(pool_env):
    _, paths, pool = pool_env
    now = datetime.now(UTC)
    with session_scope() as session:
        for index in range(6):
            session.add(
                add_block(paths, f"b{index}", plays=index, last_played=now - timedelta(hours=index))
            )
    with session_scope() as session:
        removed = pool.prune(session)
    assert len(removed) == 3
    with session_scope() as session:
        assert pool.stats(session).ready == 3


def test_prune_never_deletes_a_pinned_block(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "keep", plays=99, pinned=True))
        for index in range(5):
            session.add(add_block(paths, f"b{index}", plays=index))
    with session_scope() as session:
        removed = pool.prune(session)
    assert "keep" not in removed
    with session_scope() as session:
        assert session.get(Block, "keep") is not None


def test_prune_deletes_the_files_too(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        for index in range(5):
            session.add(add_block(paths, f"b{index}", plays=index))
    with session_scope() as session:
        removed = pool.prune(session)
    for block_id in removed:
        assert not (paths.blocks / f"block_{block_id}.ts").exists()


def test_prune_respects_the_disk_quota(tmp_path: Path):
    cfg = load_config(
        None,
        environ={
            "TAD_APP__DATA_DIR": str(tmp_path / "data"),
            "TAD_VISUAL__POOL__MAX_BLOCKS": "100",
            "TAD_VISUAL__POOL__MAX_DISK_GB": "0.000002",  # about 2 KB
        },
    )
    paths = Paths.from_config(cfg)
    paths.ensure()
    init_engine(paths.db_file)
    create_tables()
    pool = BlockPool(cfg, paths, EncodeProfile._build(cfg, "libx264"))
    try:
        with session_scope() as session:
            for index in range(5):
                session.add(add_block(paths, f"b{index}", plays=index, size=1024))
        with session_scope() as session:
            pool.prune(session)
        with session_scope() as session:
            assert pool.stats(session).size_bytes <= 2048
    finally:
        dispose()


def test_pick_prefers_the_least_played(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "cold", plays=0))
        session.add(add_block(paths, "hot", plays=200))

    counts = {"cold": 0, "hot": 0}
    rng = random.Random(3)
    with session_scope() as session:
        for _ in range(300):
            counts[pool.pick(session, 1, rng=rng)[0].id] += 1
    assert counts["cold"] > counts["hot"] * 5


def test_pick_returns_nothing_from_an_empty_pool(pool_env):
    _, _, pool = pool_env
    with session_scope() as session:
        assert pool.pick(session, 3) == []


def test_pick_does_not_repeat_within_one_call(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        for index in range(4):
            session.add(add_block(paths, f"b{index}"))
    with session_scope() as session:
        chosen = pool.pick(session, 4)
    assert len({b.id for b in chosen}) == 4


def test_mark_played_updates_the_counter(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "x"))
    with session_scope() as session:
        pool.mark_played(session, "x")
    with session_scope() as session:
        block = session.get(Block, "x")
        assert block.play_count == 1
        assert block.last_played_at is not None


def test_below_minimum_and_at_capacity_flags(pool_env):
    _, paths, pool = pool_env
    with session_scope() as session:
        assert pool.stats(session).below_minimum
    with session_scope() as session:
        for index in range(3):
            session.add(add_block(paths, f"b{index}"))
    with session_scope() as session:
        stats = pool.stats(session)
        assert not stats.below_minimum
        assert stats.at_capacity


def test_a_corrupt_sidecar_does_not_stop_adoption(pool_env):
    """A block with an unreadable sidecar is still a playable block."""
    _, paths, pool = pool_env
    (paths.blocks / "block_broken.ts").write_bytes(b"0" * 256)
    (paths.blocks / "block_broken.json").write_text("{ this is not json", encoding="utf-8")
    with session_scope() as session:
        assert pool.reconcile(session)["added"] == 1
    with session_scope() as session:
        assert session.get(Block, "broken") is not None


def test_selected_blocks_stay_readable_after_the_session_closes(pool_env):
    """The feeder picks a block in one scope and uses it in another."""
    _, paths, pool = pool_env
    with session_scope() as session:
        session.add(add_block(paths, "z"))
    with session_scope() as session:
        chosen = pool.pick(session, 1)
    assert chosen[0].id == "z"
    assert chosen[0].relpath.endswith("block_z.ts")
