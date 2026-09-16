"""Runs every stage for each document in a source. One failing document never stops the batch."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from mayakb.core import registry
from mayakb.core.contracts import SourceRef
from mayakb.ingest.jobs import (
    IngestContext,
    JobFailed,
    JobResult,
    chunk_document,
    embed_chunks,
    extract_entities,
    index_chunks,
    read_source,
    understand_document,
)


class DocumentOutcome(BaseModel):
    source_uri: str
    status: Literal["succeeded", "skipped", "failed"]
    failed_stage: str | None = None
    error: str | None = None
    jobs: list[JobResult] = Field(default_factory=list)


class IngestReport(BaseModel):
    kb: str
    documents: list[DocumentOutcome] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        found = Counter(d.status for d in self.documents)
        return {status: found.get(status, 0) for status in ("succeeded", "skipped", "failed")}


def ingest_document(ref: SourceRef, *, ctx: IngestContext) -> DocumentOutcome:
    jobs: list[JobResult] = []
    try:
        jobs.append(read_source(ref, ctx=ctx))
        jobs.append(understand_document(jobs[-1].output_id, ctx=ctx))
        jobs.append(chunk_document(jobs[-1].output_id, ctx=ctx))
        chunk_set_id = jobs[-1].output_id
        with ThreadPoolExecutor(max_workers=2) as pool:
            embedding = pool.submit(embed_chunks, chunk_set_id, ctx=ctx)
            entities = pool.submit(extract_entities, chunk_set_id, ctx=ctx) if ctx.kb.ner else None
            vector_result = embedding.result()
            entity_result = entities.result() if entities else None
        jobs.append(vector_result)
        if entity_result:
            jobs.append(entity_result)
        jobs.append(index_chunks(chunk_set_id, vector_result.output_id,
                                 entity_result.output_id if entity_result else None, ctx=ctx))
    except JobFailed as exc:
        return DocumentOutcome(source_uri=ref.uri, status="failed", failed_stage=exc.stage,
                               error=f"{type(exc.cause).__name__}: {exc.cause}", jobs=jobs)
    status = "skipped" if all(job.status == "skipped" for job in jobs) else "succeeded"
    return DocumentOutcome(source_uri=ref.uri, status=status, jobs=jobs)


def ingest(ctx: IngestContext, refs: Iterable[SourceRef] | None = None) -> IngestReport:
    """Ingest every document the KB's connector lists (or the given refs)."""
    if refs is None:
        source = ctx.kb.source
        refs = registry.create("connector", source.name, **source.params).list()
    report = IngestReport(kb=ctx.kb.kb)
    for ref in refs:
        report.documents.append(ingest_document(ref, ctx=ctx))
    return report
