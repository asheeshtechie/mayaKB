"""extract_entities, embed_chunks, index_chunks and run_stage."""

import pytest

from mayakb.core.contracts import EntitySet, SourceRef, VectorSet
from mayakb.ingest.jobs import (
    IngestContext,
    JobFailed,
    chunk_document,
    embed_chunks,
    extract_entities,
    index_chunks,
    read_source,
    run_stage,
    understand_document,
)
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.joblog import InMemoryJobLog
from mayakb.storage.search.base import Filters
from mayakb.storage.search.memory import InMemorySearchBackend
from tests.fakes import make_kb

URI = "mem://docs/policy.md"
POLICY = "# Refunds\n\nAcme refunds take 30 days.\n\n# Shipping\n\nOrders ship from Berlin."


@pytest.fixture
def ctx(tmp_path):
    return IngestContext(kb=make_kb({URI: POLICY}), store=FileArtifactStore(tmp_path), joblog=InMemoryJobLog(),
                         backend=InMemorySearchBackend())


def chunked(ctx) -> str:
    raw_id = read_source(SourceRef(uri=URI), ctx=ctx).output_id
    return chunk_document(understand_document(raw_id, ctx=ctx).output_id, ctx=ctx).output_id


def test_extract_entities(ctx):
    result = extract_entities(chunked(ctx), ctx=ctx)
    entity_set = ctx.store.read_model("test-kb", "entities", result.output_id, EntitySet)
    assert {e.normalized for e in entity_set.entities} >= {"acme", "berlin"}


def test_embed_chunks_writes_vectors_in_chunk_order(ctx):
    chunk_set_id = chunked(ctx)
    result = embed_chunks(chunk_set_id, ctx=ctx)
    vector_set = ctx.store.read_model("test-kb", "vectors", result.output_id, VectorSet)
    vectors = ctx.store.read_array("test-kb", "vectors", result.output_id)
    assert vector_set.dim == 64
    assert vector_set.model == "hash-64"
    assert vectors.shape == (len(vector_set.chunk_ids), 64)


def test_index_chunks_loads_rows_with_entities_and_skips_reruns(ctx):
    chunk_set_id = chunked(ctx)
    vector_id = embed_chunks(chunk_set_id, ctx=ctx).output_id
    entity_id = extract_entities(chunk_set_id, ctx=ctx).output_id

    first = index_chunks(chunk_set_id, vector_id, entity_id, ctx=ctx)
    assert first.status == "succeeded"
    index = ctx.backend.find_index("test-kb", "hash", "hash-64")
    hits = ctx.backend.keyword_search("berlin", 5, Filters(entities=["berlin"]), index)
    assert [h.section_path for h in hits] == [["Shipping"]]
    assert hits[0].source_uri == URI

    assert index_chunks(chunk_set_id, vector_id, entity_id, ctx=ctx).status == "skipped"


def test_index_rejects_vectors_from_another_chunk_set(ctx):
    chunk_set_id = chunked(ctx)
    vector_id = embed_chunks(chunk_set_id, ctx=ctx).output_id
    ctx.kb = make_kb({URI: POLICY}, chunk={"strategy": "fixed", "max_tokens": 4, "overlap_tokens": 0})
    other_chunks = chunked(ctx)
    with pytest.raises(JobFailed, match="does not belong"):
        index_chunks(other_chunks, vector_id, ctx=ctx)


def test_empty_document_indexes_nothing(ctx):
    ctx.kb = make_kb({URI: ""})
    chunk_set_id = chunked(ctx)
    vector_id = embed_chunks(chunk_set_id, ctx=ctx).output_id
    assert ctx.store.read_model("test-kb", "vectors", vector_id, VectorSet).dim == 0
    assert index_chunks(chunk_set_id, vector_id, ctx=ctx).status == "succeeded"
    assert ctx.backend.list_indexes("test-kb") == []


def test_run_stage_dispatches_by_name(ctx):
    raw_id = run_stage("read_source", [URI], ctx=ctx).output_id
    doc_id = run_stage("understand_document", [raw_id], ctx=ctx).output_id
    chunk_set_id = run_stage("chunk_document", [doc_id], ctx=ctx).output_id
    vector_id = run_stage("embed_chunks", [chunk_set_id], ctx=ctx).output_id
    entity_id = run_stage("extract_entities", [chunk_set_id], ctx=ctx).output_id
    assert run_stage("index_chunks", [chunk_set_id, vector_id, entity_id], ctx=ctx).status == "succeeded"
    with pytest.raises(ValueError, match="unknown stage"):
        run_stage("teleport", [raw_id], ctx=ctx)
    with pytest.raises(ValueError):
        run_stage("chunk_document", [], ctx=ctx)
