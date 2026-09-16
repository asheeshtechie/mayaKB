"""The six ingestion jobs. Each reads its inputs from the ArtifactStore and writes its output there.

A job is skipped when its output already exists for the same inputs, implementation and params.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel

from mayakb.core import registry
from mayakb.core.config import ImplConfig, KBConfig
from mayakb.core.contracts import Chunk, ChunkSet, DocModel, EntitySet, RawDocument, SourceRef, VectorSet
from mayakb.core.ids import stable_id
from mayakb.storage.artifacts import ArtifactStore
from mayakb.storage.joblog import JobLog, JobRecord
from mayakb.storage.search.base import IndexRow, SearchBackend

STAGES = (
    "read_source",
    "understand_document",
    "chunk_document",
    "extract_entities",
    "embed_chunks",
    "index_chunks",
)


@dataclass
class IngestContext:
    kb: KBConfig
    store: ArtifactStore
    joblog: JobLog
    backend: SearchBackend | None = None


class JobResult(BaseModel):
    job_id: str
    stage: str
    status: Literal["succeeded", "skipped"]
    output_id: str


class JobFailed(RuntimeError):
    def __init__(self, job_id: str, stage: str, cause: BaseException):
        super().__init__(f"{stage} failed (job {job_id}): {type(cause).__name__}: {cause}")
        self.job_id = job_id
        self.stage = stage
        self.cause = cause


def _run(
    ctx: IngestContext,
    *,
    stage: str,
    input_ids: list[str],
    impl: ImplConfig,
    version: str,
    check: Callable[[JobRecord | None], str | None],
    work: Callable[[], str],
) -> JobResult:
    """Log the job, skip it if `check` finds its output, otherwise run `work` (returns the output id)."""
    kb = ctx.kb.kb
    job_id = stable_id("job", kb, stage, input_ids, impl.name, version, impl.params)
    previous = ctx.joblog.get(job_id)
    ctx.joblog.start(
        job_id=job_id,
        kb=kb,
        stage=stage,
        input_ids=input_ids,
        implementation=f"{impl.name}@{version}",
        params_hash=stable_id(impl.params),
    )
    try:
        existing = check(previous)
        if existing is not None:
            ctx.joblog.finish(job_id, status="skipped", output_id=existing)
            return JobResult(job_id=job_id, stage=stage, status="skipped", output_id=existing)
        output_id = work()
    except Exception as exc:
        ctx.joblog.fail(job_id, f"{type(exc).__name__}: {exc}")
        raise JobFailed(job_id, stage, exc) from exc
    ctx.joblog.finish(job_id, status="succeeded", output_id=output_id)
    return JobResult(job_id=job_id, stage=stage, status="succeeded", output_id=output_id)


def _artifact_job(
    ctx: IngestContext,
    *,
    stage: str,
    input_ids: list[str],
    kind: str,
    impl: ImplConfig,
    registry_kind: str,
    produce: Callable[[str], None],
) -> JobResult:
    """A job whose output is one artifact with a deterministic id."""
    version = registry.get(registry_kind, impl.name).version
    output_id = stable_id(stage, input_ids, impl.name, version, impl.params)

    def work() -> str:
        produce(output_id)
        return output_id

    return _run(
        ctx,
        stage=stage,
        input_ids=input_ids,
        impl=impl,
        version=version,
        check=lambda _: output_id if ctx.store.exists(ctx.kb.kb, kind, output_id) else None,
        work=work,
    )


def read_source(ref: SourceRef, *, ctx: IngestContext) -> JobResult:
    """Fetch a document. Skipped when the source's etag hasn't changed since the last fetch."""
    impl, kb = ctx.kb.source, ctx.kb.kb
    version = registry.get("connector", impl.name).version
    map_key = stable_id("source", ref.uri)

    def check(_: JobRecord | None) -> str | None:
        seen = ctx.store.read_json(kb, "source_map", map_key)
        unchanged = seen and ref.etag and seen.get("etag") == ref.etag
        return seen["raw_id"] if unchanged and ctx.store.exists(kb, "raw", seen["raw_id"]) else None

    def work() -> str:
        connector = registry.create("connector", impl.name, **impl.params)
        raw, data = connector.fetch(ref)
        ctx.store.write_bytes(kb, "raw", raw.id, data)
        ctx.store.write_model(kb, "raw", raw.id, raw)  # written last: its presence means "complete"
        ctx.store.write_json(kb, "source_map", map_key, {"uri": ref.uri, "etag": ref.etag, "raw_id": raw.id})
        return raw.id

    return _run(ctx, stage="read_source", input_ids=[ref.uri], impl=impl, version=version, check=check, work=work)


def understand_document(raw_id: str, *, ctx: IngestContext) -> JobResult:
    impl, kb = ctx.kb.parse, ctx.kb.kb

    def produce(output_id: str) -> None:
        raw = ctx.store.read_model(kb, "raw", raw_id, RawDocument)
        data = ctx.store.read_bytes(kb, "raw", raw_id)
        parser = registry.create("parser", impl.name, **impl.params)
        ctx.store.write_model(kb, "doc_model", output_id, parser.parse(raw, data))

    return _artifact_job(ctx, stage="understand_document", input_ids=[raw_id], kind="doc_model", impl=impl,
                         registry_kind="parser", produce=produce)


def chunk_document(doc_model_id: str, *, ctx: IngestContext) -> JobResult:
    impl, kb = ctx.kb.chunk, ctx.kb.kb

    def produce(output_id: str) -> None:
        doc = ctx.store.read_model(kb, "doc_model", doc_model_id, DocModel)
        chunker = registry.create("chunker", impl.name, **impl.params)
        chunks = [
            Chunk(**piece.model_dump(), chunk_id=stable_id(output_id, position), doc_id=doc.doc_id,
                  position=position, acl=doc.acl)
            for position, piece in enumerate(chunker.chunk(doc))
        ]
        chunk_set = ChunkSet(id=output_id, doc_id=doc.doc_id, source_uri=doc.source_uri, title=doc.title,
                             strategy=impl.name, chunks=chunks)
        ctx.store.write_model(kb, "chunks", output_id, chunk_set)

    return _artifact_job(ctx, stage="chunk_document", input_ids=[doc_model_id], kind="chunks", impl=impl,
                         registry_kind="chunker", produce=produce)


def extract_entities(chunk_set_id: str, *, ctx: IngestContext) -> JobResult:
    impl, kb = ctx.kb.ner, ctx.kb.kb
    if impl is None:
        raise ValueError(f"knowledge base {kb!r} has no ner section")

    def produce(output_id: str) -> None:
        chunk_set = ctx.store.read_model(kb, "chunks", chunk_set_id, ChunkSet)
        extractor = registry.create("ner", impl.name, **impl.params)
        entities = extractor.extract(chunk_set.chunks) if chunk_set.chunks else []
        entity_set = EntitySet(id=output_id, chunk_set_id=chunk_set_id, extractor=impl.name, entities=entities)
        ctx.store.write_model(kb, "entities", output_id, entity_set)

    return _artifact_job(ctx, stage="extract_entities", input_ids=[chunk_set_id], kind="entities", impl=impl,
                         registry_kind="ner", produce=produce)


def embed_chunks(chunk_set_id: str, *, ctx: IngestContext) -> JobResult:
    impl, kb = ctx.kb.embed, ctx.kb.kb

    def produce(output_id: str) -> None:
        chunk_set = ctx.store.read_model(kb, "chunks", chunk_set_id, ChunkSet)
        embedder = registry.create("embedder", impl.name, **impl.params)
        texts = [c.text for c in chunk_set.chunks]
        vectors = np.asarray(embedder.embed_documents(texts) if texts else [], dtype=np.float32)
        dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        ctx.store.write_array(kb, "vectors", output_id, vectors)
        ctx.store.write_model(kb, "vectors", output_id, VectorSet(
            id=output_id, chunk_set_id=chunk_set_id, embedder=impl.name, model=embedder.model, dim=dim,
            chunk_ids=[c.chunk_id for c in chunk_set.chunks],
        ))

    return _artifact_job(ctx, stage="embed_chunks", input_ids=[chunk_set_id], kind="vectors", impl=impl,
                         registry_kind="embedder", produce=produce)


def index_chunks(
    chunk_set_id: str, vector_set_id: str, entity_set_id: str | None = None, *, ctx: IngestContext
) -> JobResult:
    """Load chunks, vectors and entities into the search backend.

    The output lives in the database, not the artifact store, so "already done" is read from the
    job log (which lives in the same database and moves with it).
    """
    if ctx.backend is None:
        raise ValueError("index_chunks needs a search backend")
    impl, kb, backend = ctx.kb.index, ctx.kb.kb, ctx.backend
    input_ids = [chunk_set_id, vector_set_id] + ([entity_set_id] if entity_set_id else [])
    output_id = stable_id("index_chunks", input_ids, impl.name, backend.version, impl.params)

    def check(previous: JobRecord | None) -> str | None:
        done = previous and previous.status in ("succeeded", "skipped") and previous.output_id == output_id
        return output_id if done else None

    def work() -> str:
        chunk_set = ctx.store.read_model(kb, "chunks", chunk_set_id, ChunkSet)
        vector_set = ctx.store.read_model(kb, "vectors", vector_set_id, VectorSet)
        vectors = ctx.store.read_array(kb, "vectors", vector_set_id)
        if vector_set.chunk_ids != [c.chunk_id for c in chunk_set.chunks]:
            raise ValueError(f"vector set {vector_set_id} does not belong to chunk set {chunk_set_id}")
        entities: dict[str, set[str]] = defaultdict(set)
        if entity_set_id:
            for entity in ctx.store.read_model(kb, "entities", entity_set_id, EntitySet).entities:
                entities[entity.chunk_id].add(entity.normalized)
        if chunk_set.chunks:
            index = backend.ensure_index(kb, vector_set.embedder, vector_set.model, vector_set.dim,
                                         impl.params.get("text_search_language", "english"))
            backend.upsert([
                IndexRow(
                    chunk_id=c.chunk_id, doc_id=c.doc_id, position=c.position, text=c.text,
                    embedding=vectors[i].tolist(), source_uri=chunk_set.source_uri, title=chunk_set.title,
                    page_start=c.page_start, page_end=c.page_end, section_path=c.section_path,
                    entities=sorted(entities[c.chunk_id]), acl=c.acl, metadata=c.metadata,
                )
                for i, c in enumerate(chunk_set.chunks)
            ], index)
        return output_id

    return _run(ctx, stage="index_chunks", input_ids=input_ids, impl=impl, version=backend.version,
                check=check, work=work)


def run_stage(stage: str, input_ids: list[str], *, ctx: IngestContext) -> JobResult:
    """Run one job by stage name, e.g. from the CLI or when retrying a failed job."""
    if stage == "read_source":
        return read_source(SourceRef(uri=input_ids[0]), ctx=ctx)
    single = {
        "understand_document": understand_document,
        "chunk_document": chunk_document,
        "extract_entities": extract_entities,
        "embed_chunks": embed_chunks,
    }
    if stage in single:
        if len(input_ids) != 1:
            raise ValueError(f"{stage} takes exactly one input id")
        return single[stage](input_ids[0], ctx=ctx)
    if stage == "index_chunks":
        if len(input_ids) not in (2, 3):
            raise ValueError("index_chunks takes a chunk set id, a vector set id and optionally an entity set id")
        return index_chunks(*input_ids, ctx=ctx)
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
