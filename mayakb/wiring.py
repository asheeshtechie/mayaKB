"""Builds the real runtime from environment settings: Postgres, filesystem artifacts, KB config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg_pool import ConnectionPool

from mayakb.core.config import KBConfig, Settings
from mayakb.ingest.jobs import IngestContext
from mayakb.retrieval.search import Searcher, SearchOptions, SearchResults
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.db import DatabaseUnavailable, make_pool
from mayakb.storage.joblog import JobLog, JobRecord, PostgresJobLog
from mayakb.storage.search.base import Filters, SearchBackend
from mayakb.storage.search.postgres import PostgresSearchBackend


@dataclass
class Runtime:
    settings: Settings
    kb: KBConfig
    store: FileArtifactStore
    joblog: JobLog
    backend: SearchBackend
    pool: ConnectionPool | None = None

    def context(self) -> IngestContext:
        return IngestContext(kb=self.kb, store=self.store, joblog=self.joblog, backend=self.backend)

    def searcher(self) -> Searcher:
        return Searcher.from_config(self.kb, self.backend)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()


def load_kb(name: str, settings: Settings) -> KBConfig:
    path = settings.kb_config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no config for knowledge base {name!r} at {path}")
    kb = KBConfig.from_yaml(path)
    if kb.kb != name:
        raise ValueError(f"{path} declares kb {kb.kb!r}; expected {name!r}")
    return kb


def open_pool(settings: Settings) -> ConnectionPool:
    if not settings.database_url:
        raise DatabaseUnavailable("MAYAKB_DATABASE_URL is not set")
    return make_pool(settings.database_url)


def build_runtime(kb_name: str, settings: Settings | None = None, pool: ConnectionPool | None = None) -> Runtime:
    settings = settings or Settings.from_env()
    kb = load_kb(kb_name, settings)
    if kb.index.name != "postgres":
        raise ValueError(f"index backend {kb.index.name!r} is not available; use 'postgres'")
    pool = pool or open_pool(settings)
    try:
        return Runtime(
            settings=settings,
            kb=kb,
            store=FileArtifactStore(settings.data_dir),
            joblog=PostgresJobLog(pool),
            backend=PostgresSearchBackend(pool),
            pool=pool,
        )
    except Exception:
        pool.close()
        raise


def find_job(job_id: str, settings: Settings | None = None) -> tuple[JobRecord, ConnectionPool]:
    """The job's record and an open pool the caller can reuse for build_runtime."""
    pool = open_pool(settings or Settings.from_env())
    record = PostgresJobLog(pool).get(job_id)
    if record is None:
        pool.close()
        raise LookupError(f"no job {job_id!r}")
    return record, pool


def search(
    query: str,
    kb: str,
    filters: Filters | dict[str, Any] | None = None,
    options: SearchOptions | None = None,
    principals: list[str] | None = None,
    settings: Settings | None = None,
) -> SearchResults:
    """One-shot search using environment settings. Long-running callers should keep a Runtime."""
    runtime = build_runtime(kb, settings)
    try:
        return runtime.searcher().search(query, filters=filters, options=options, principals=principals)
    finally:
        runtime.close()
