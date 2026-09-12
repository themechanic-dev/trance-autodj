"""SQLite engine and session handling.

SQLite is used deliberately: one file, no daemon, nothing to administer for a
system meant to run unattended for months. The settings below are what make
that safe when a web process, a generator and a streamer all touch it at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.core.logging import get_logger

log = get_logger(__name__)

_engine: Engine | None = None


def _configure_connection(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
    cursor = dbapi_connection.cursor()
    # WAL lets the readers (web) work while a writer (generator) commits.
    cursor.execute("PRAGMA journal_mode=WAL")
    # NORMAL is the right trade for WAL: durable across process crashes, and
    # only at risk from a power cut, which for a media pool costs nothing.
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    # Wait rather than raise "database is locked" during a concurrent write.
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


def init_engine(db_file: Path, *, echo: bool = False) -> Engine:
    # A module-level singleton is deliberate: one engine per process, shared
    # by the request handlers and the background workers.
    global _engine  # noqa: PLW0603
    db_file.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(
        f"sqlite:///{db_file}",
        echo=echo,
        connect_args={"check_same_thread": False, "timeout": 10.0},
    )
    event.listen(_engine, "connect", _configure_connection)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("the database engine has not been initialised")
    return _engine


# SQLite types for the Python types the models actually use.
_SQL_TYPES = {"INTEGER", "REAL", "TEXT", "BOOLEAN", "VARCHAR", "FLOAT", "DATETIME", "JSON"}


def create_tables() -> None:
    # Importing the module registers every table on SQLModel.metadata.
    from app.models import entities  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
    added = _add_missing_columns()
    if added:
        log.info("database schema migrated", extra={"added_columns": added})
    log.info("database schema is up to date")


def _add_missing_columns() -> list[str]:
    """Add columns the models have gained since the database was created.

    ``create_all`` only ever creates missing *tables*; a table that already
    exists is left exactly as it is. For a system meant to run unattended for
    months, that means every new field would break it on upgrade with an
    "no such column" error at the first query. Adding them here keeps an
    upgrade to "pull the new image and restart".

    Only additive, and only for simple columns — anything that needs data
    moved deserves a real migration and a human.
    """
    from sqlalchemy import inspect, text

    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                try:
                    type_sql = column.type.compile(engine.dialect)
                except Exception:
                    log.warning(
                        "cannot add %s.%s automatically; migrate by hand",
                        table.name,
                        column.name,
                    )
                    continue
                if type_sql.split("(")[0].upper() not in _SQL_TYPES:
                    log.warning(
                        "skipping %s.%s (%s): not a simple column",
                        table.name,
                        column.name,
                        type_sql,
                    )
                    continue

                default = "NULL"
                if not column.nullable:
                    # A NOT NULL column needs something for the existing rows.
                    default = (
                        "0"
                        if type_sql.upper().startswith(("INT", "REAL", "FLOAT", "BOOL"))
                        else "''"
                    )
                connection.execute(
                    text(
                        f"ALTER TABLE {table.name} "
                        f"ADD COLUMN {column.name} {type_sql} DEFAULT {default}"
                    )
                )
                added.append(f"{table.name}.{column.name}")

    return added


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional session: commit on success, roll back on failure.

    ``expire_on_commit`` is off deliberately. With SQLAlchemy's default, every
    attribute of every object is invalidated at commit, so reading ``block.id``
    after the ``with`` block raises DetachedInstanceError. Callers that hand a
    selected block on to something else — the streamer's feeder does exactly
    that — would each need their own workaround. Rows here are short-lived
    snapshots, not long-lived shared state, so keeping them readable is both
    safe and far less surprising.
    """
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def dispose() -> None:
    global _engine  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
        _engine = None


__all__ = [
    "create_tables",
    "dispose",
    "get_engine",
    "get_session",
    "init_engine",
    "session_scope",
]
