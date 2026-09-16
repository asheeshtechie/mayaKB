"""Postgres connection pool and schema migrations."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIN_PGVECTOR = (0, 8, 0)  # iterative index scans keep filtered vector searches from coming back short
_MIGRATION_LOCK = 7_272_727  # advisory lock key: two containers never migrate at the same time


class DatabaseUnavailable(RuntimeError):
    pass


def redact(url: str) -> str:
    """Hide the password in a connection URL."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", url)


def apply_migrations(database_url: str) -> list[str]:
    """Apply migrations/*.sql not yet recorded in schema_migrations. Returns the ones applied now."""
    try:
        conn = psycopg.connect(database_url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailable(
            f"cannot connect to Postgres at MAYAKB_DATABASE_URL={redact(database_url)}: {exc}"
        ) from exc
    applied: list[str] = []
    with conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.stem in done:
                    continue
                with conn.transaction():
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                applied.append(path.stem)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))
    return applied


def pgvector_version(conn: psycopg.Connection) -> tuple[int, ...] | None:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if row is None:
        return None
    return tuple(int(part) for part in re.findall(r"\d+", row[0])[:3])


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()


def make_pool(database_url: str, *, min_size: int = 1, max_size: int = 10, timeout: float = 10.0) -> ConnectionPool:
    pool = ConnectionPool(database_url, min_size=min_size, max_size=max_size, configure=_configure, open=False)
    try:
        pool.open(wait=True, timeout=timeout)
    except PoolTimeout as exc:
        pool.close()
        raise DatabaseUnavailable(
            f"cannot connect to Postgres at MAYAKB_DATABASE_URL={redact(database_url)} "
            "(is the db container healthy, and has `mayakb db init` run?)"
        ) from exc
    return pool
