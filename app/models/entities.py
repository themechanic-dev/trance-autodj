"""SQLModel tables. SQLite is the whole database — no external service.

Two rules shape these models:

* The filesystem is the source of truth for media. Rows describe files; they
  never *are* the file. A missing file marks its row, it does not delete it,
  so a temporarily unmounted disk cannot wipe the library.
* Nothing secret is stored here. The stream key lives in the encrypted secret
  store (see :mod:`app.core.security`), not in this database.
"""

import enum
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class BlockSource(str, enum.Enum):
    AI = "ai"
    PROCEDURAL = "procedural"
    MIXED = "mixed"
    FALLBACK = "fallback"


class BlockStatus(str, enum.Enum):
    BUILDING = "building"
    READY = "ready"
    REJECTED = "rejected"
    MISSING = "missing"


class JobKind(str, enum.Enum):
    BLOCK = "block"
    CLIP = "clip"
    IMAGE = "image"
    ANALYSE = "analyse"
    #: A composed music track. Stored as a string like the rest, so adding it
    #: needs no migration.
    TRACK = "track"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlaylistMode(str, enum.Enum):
    SHUFFLE = "shuffle"
    SEQUENTIAL = "sequential"
    RANDOM = "random"


# --------------------------------------------------------------------------
# music
# --------------------------------------------------------------------------


class Track(SQLModel, table=True):
    __tablename__ = "tracks"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Relative to Paths.music, so the library survives a data_dir move.
    relpath: str = Field(index=True, unique=True)
    filename: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: Optional[int] = None
    duration_s: float = 0.0
    bitrate_k: int = 0
    sample_rate: int = 0
    size_bytes: int = 0
    # Content hash, for duplicate detection on upload.
    sha256: str = Field(default="", index=True)
    bpm: Optional[float] = None
    # Seconds from the start of the file to the first beat, modulo one beat.
    beat_offset_s: Optional[float] = None
    bpm_confidence: float = 0.0
    analysis_method: str = ""
    musical_key: str = ""
    replaygain_db: Optional[float] = None
    artwork_relpath: str = ""
    play_count: int = Field(default=0, index=True)
    last_played_at: Optional[datetime] = None
    added_at: datetime = Field(default_factory=utcnow)
    analysed_at: datetime | None = None
    # Set when the file has gone; the row is kept so playlists do not silently
    # lose entries when a mount comes back.
    missing: bool = Field(default=False, index=True)
    # Marked by the operator, usually because something was claimed against
    # it. Flagging rather than deleting on the spot: a claim arrives while
    # the station is on air, and taking a track out from under Liquidsoap in
    # the middle of a broadcast is how you get silence.
    flagged: bool = Field(default=False, index=True)
    flag_note: str = ""

    # Deleting a track deletes its playlist entries with it. Without the
    # cascade SQLAlchemy tries to disown the children instead, by setting
    # playlist_items.track_id to NULL — a NOT NULL column — so the commit
    # fails. That commit happens in the request's teardown, *after* the
    # endpoint has answered 200 and after the file on disk is already gone:
    # the caller is told the track was deleted, the audio file really is
    # deleted, and the row survives for ever.
    items: List["PlaylistItem"] = Relationship(
        back_populates="track",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Playlist(SQLModel, table=True):
    __tablename__ = "playlists"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    mode: PlaylistMode = Field(default=PlaylistMode.SHUFFLE)
    # None means "use the global audio.crossfade.duration_s".
    crossfade_s: Optional[float] = None
    is_active: bool = Field(default=False, index=True)
    # Where this playlist sits in the rotation, or None if it is not in it.
    # The rotation is an order the operator arranges by hand; `is_active` is
    # the separate question of which single playlist plays when the station is
    # not rotating at all.
    rotation_position: Optional[int] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    items: List["PlaylistItem"] = Relationship(
        back_populates="playlist",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "PlaylistItem.position",
        },
    )


class PlaylistItem(SQLModel, table=True):
    __tablename__ = "playlist_items"
    __table_args__ = (UniqueConstraint("playlist_id", "position", name="uq_playlist_position"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    playlist_id: int = Field(foreign_key="playlists.id", index=True, ondelete="CASCADE")
    track_id: int = Field(foreign_key="tracks.id", index=True, ondelete="CASCADE")
    position: int = 0

    playlist: Optional[Playlist] = Relationship(back_populates="items")
    track: Optional[Track] = Relationship(back_populates="items")


# --------------------------------------------------------------------------
# visuals
# --------------------------------------------------------------------------


class Block(SQLModel, table=True):
    """One ready-to-stream .ts segment with its crossfades already baked in."""

    __tablename__ = "blocks"

    id: str = Field(primary_key=True)  # uuid4 hex, also the filename stem
    relpath: str = Field(index=True)
    duration_s: float = 0.0
    size_bytes: int = 0
    source: BlockSource = Field(default=BlockSource.PROCEDURAL, index=True)
    status: BlockStatus = Field(default=BlockStatus.BUILDING, index=True)
    reject_reason: str = ""
    # Identity of the encode profile that produced it. A block whose
    # fingerprint no longer matches the configured profile cannot be fed to a
    # "-c:v copy" stream, so the pool refuses it.
    profile_fingerprint: str = Field(default="", index=True)
    thumbnail_relpath: str = ""
    # Free-form provenance: generators used, prompts, seeds, transitions.
    # Kept as JSON text because it is read by humans, not queried.
    metadata_json: str = "{}"
    play_count: int = Field(default=0, index=True)
    last_played_at: Optional[datetime] = None
    pinned: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)


class PlayLogEntry(SQLModel, table=True):
    """What played, and when — the answer to a claim that carries a timestamp.

    The track's own `play_count` and `last_played_at` say how often and how
    recently; neither can answer "what was on air at 23:14 on Tuesday", which
    is the only question a copyright notice actually asks.

    The name, the artist and the path are copied in rather than looked up
    through the relation on purpose: the whole point of the record is that it
    still answers after the track has been deleted, which is exactly what an
    operator does the moment they find out what was claimed.
    """

    __tablename__ = "play_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: Optional[int] = Field(
        default=None, foreign_key="tracks.id", index=True, ondelete="SET NULL"
    )
    relpath: str = Field(default="", index=True)
    title: str = ""
    artist: str = ""
    started_at: datetime = Field(default_factory=utcnow, index=True)
    #: Left open while the track is playing, so "what is on now" and "what was
    #: on then" are the same query.
    ended_at: Optional[datetime] = Field(default=None, index=True)


class GenerationJob(SQLModel, table=True):
    __tablename__ = "generation_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    kind: JobKind = Field(default=JobKind.BLOCK, index=True)
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    progress: float = 0.0
    step: str = ""
    params_json: str = "{}"
    result_id: str = ""
    error: str = ""
    created_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# --------------------------------------------------------------------------
# settings written by the UI
# --------------------------------------------------------------------------


class Setting(SQLModel, table=True):
    """Overrides the dashboard writes at runtime.

    Applied on top of config.yaml but below environment variables, so an
    operator pinning a value via the environment is never overruled from the
    web UI.
    """

    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value_json: str = "null"
    updated_at: datetime = Field(default_factory=utcnow)


__all__ = [
    "Block",
    "BlockSource",
    "BlockStatus",
    "GenerationJob",
    "JobKind",
    "JobStatus",
    "Playlist",
    "PlaylistItem",
    "PlaylistMode",
    "Setting",
    "Track",
    "utcnow",
]
