"""Database engine/session wiring that works on both SQLite and PostgreSQL."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    # Normalised so provider-issued postgres:// URLs resolve to the psycopg
    # driver rather than failing at connect time.
    url = settings.normalised_database_url()
    kwargs: dict = {"echo": settings.SQL_ECHO, "future": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    else:
        # pool_recycle guards against managed Postgres providers dropping idle
        # connections: without it the first request after a quiet period fails
        # with "server closed the connection unexpectedly".
        kwargs.update(
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=280,
            pool_timeout=30,
        )

    eng = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _rec):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return eng


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one transactional session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Import models first so metadata is populated."""
    import app.models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(bind=engine)
