"""Job log: one row per job, updated as the job runs, fails, or is skipped."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Literal, Protocol

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

JobStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class JobRecord(BaseModel):
    job_id: str
    kb: str
    stage: str
    input_ids: list[str]
    output_id: str | None = None
    implementation: str
    params_hash: str
    status: JobStatus
    attempts: int = 0
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobLog(Protocol):
    def start(
        self, *, job_id: str, kb: str, stage: str, input_ids: list[str], implementation: str, params_hash: str
    ) -> None: ...
    def finish(self, job_id: str, *, status: Literal["succeeded", "skipped"], output_id: str) -> None: ...
    def fail(self, job_id: str, error: str) -> None: ...
    def get(self, job_id: str) -> JobRecord | None: ...
    def list(self, kb: str, status: JobStatus | None = None, limit: int = 100) -> list[JobRecord]: ...


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryJobLog:
    """For tests and single-process experiments."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def start(self, *, job_id, kb, stage, input_ids, implementation, params_hash) -> None:
        with self._lock:
            previous = self._jobs.get(job_id)
            self._jobs[job_id] = JobRecord(
                job_id=job_id,
                kb=kb,
                stage=stage,
                input_ids=list(input_ids),
                output_id=previous.output_id if previous else None,
                implementation=implementation,
                params_hash=params_hash,
                status="running",
                attempts=(previous.attempts if previous else 0) + 1,
                started_at=_now(),
            )

    def finish(self, job_id, *, status, output_id) -> None:
        with self._lock:
            record = self._jobs[job_id]
            self._jobs[job_id] = record.model_copy(
                update={"status": status, "output_id": output_id, "error": None, "finished_at": _now()}
            )

    def fail(self, job_id, error) -> None:
        with self._lock:
            record = self._jobs[job_id]
            self._jobs[job_id] = record.model_copy(update={"status": "failed", "error": error, "finished_at": _now()})

    def get(self, job_id) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.model_copy() if record else None

    def list(self, kb, status=None, limit=100) -> list[JobRecord]:
        with self._lock:
            records = [r for r in self._jobs.values() if r.kb == kb and (status is None or r.status == status)]
        records.sort(key=lambda r: r.started_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return records[:limit]


_COLUMNS = (
    "job_id, kb, stage, input_ids, output_id, implementation, params_hash, "
    "status, attempts, error, started_at, finished_at"
)


class PostgresJobLog:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def start(self, *, job_id, kb, stage, input_ids, implementation, params_hash) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, kb, stage, input_ids, implementation, params_hash,
                                  status, attempts, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'running', 1, now())
                ON CONFLICT (job_id) DO UPDATE SET
                    status = 'running', attempts = jobs.attempts + 1, error = NULL,
                    implementation = EXCLUDED.implementation, params_hash = EXCLUDED.params_hash,
                    started_at = now(), finished_at = NULL
                """,
                (job_id, kb, stage, list(input_ids), implementation, params_hash),
            )

    def finish(self, job_id, *, status, output_id) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = %s, output_id = %s, error = NULL, finished_at = now() WHERE job_id = %s",
                (status, output_id, job_id),
            )

    def fail(self, job_id, error) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = %s, finished_at = now() WHERE job_id = %s",
                (error, job_id),
            )

    def get(self, job_id) -> JobRecord | None:
        with self.pool.connection() as conn:
            row = conn.cursor(row_factory=dict_row).execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        return JobRecord(**row) if row else None

    def list(self, kb, status=None, limit=100) -> list[JobRecord]:
        query = f"SELECT {_COLUMNS} FROM jobs WHERE kb = %s"
        params: list[object] = [kb]
        if status is not None:
            query += " AND status = %s"
            params.append(status)
        query += " ORDER BY started_at DESC NULLS LAST LIMIT %s"
        params.append(limit)
        with self.pool.connection() as conn:
            rows = conn.cursor(row_factory=dict_row).execute(query, params).fetchall()
        return [JobRecord(**row) for row in rows]
