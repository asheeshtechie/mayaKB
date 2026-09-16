"""read_source, understand_document and chunk_document."""

import pytest

from mayakb.core.contracts import ChunkSet, DocModel, RawDocument, SourceRef
from mayakb.core.ids import content_id
from mayakb.ingest.jobs import (
    IngestContext,
    JobFailed,
    chunk_document,
    read_source,
    understand_document,
)
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.joblog import InMemoryJobLog
from tests.fakes import make_kb

POLICY = "# Refunds\n\nRefunds take 30 days.\n\n# Shipping\n\nOrders ship in two days."
URI = "mem://docs/policy.md"


@pytest.fixture
def ctx(tmp_path):
    return IngestContext(kb=make_kb({URI: POLICY}), store=FileArtifactStore(tmp_path), joblog=InMemoryJobLog())


def ref(text: str = POLICY) -> SourceRef:
    return SourceRef(uri=URI, etag=content_id(text.encode())[:12])


def test_read_source_stores_bytes_and_metadata(ctx):
    result = read_source(ref(), ctx=ctx)
    assert result.status == "succeeded"
    assert result.output_id == content_id(POLICY.encode())
    assert ctx.store.read_bytes("test-kb", "raw", result.output_id) == POLICY.encode()
    raw = ctx.store.read_model("test-kb", "raw", result.output_id, RawDocument)
    assert raw.source_uri == URI
    assert ctx.joblog.get(result.job_id).status == "succeeded"


def test_read_source_skips_unchanged_etag_and_refetches_changes(ctx):
    first = read_source(ref(), ctx=ctx)
    again = read_source(ref(), ctx=ctx)
    assert again.status == "skipped"
    assert again.output_id == first.output_id
    assert ctx.joblog.get(again.job_id).attempts == 2

    changed = POLICY + "\n\nNew paragraph."
    ctx.kb = make_kb({URI: changed})
    third = read_source(ref(changed), ctx=ctx)
    assert third.status == "succeeded"
    assert third.output_id == content_id(changed.encode())


def test_understand_document_keeps_content_id_as_doc_id(ctx):
    raw_id = read_source(ref(), ctx=ctx).output_id
    result = understand_document(raw_id, ctx=ctx)
    doc = ctx.store.read_model("test-kb", "doc_model", result.output_id, DocModel)
    assert doc.doc_id == raw_id
    assert doc.title == "Refunds"
    assert result.output_id != raw_id
    assert understand_document(raw_id, ctx=ctx).status == "skipped"


def test_chunk_document_assigns_ids_positions_and_acl(ctx):
    raw_id = read_source(ref(), ctx=ctx).output_id
    doc_model_id = understand_document(raw_id, ctx=ctx).output_id
    doc = ctx.store.read_model("test-kb", "doc_model", doc_model_id, DocModel)
    ctx.store.write_model("test-kb", "doc_model", doc_model_id, doc.model_copy(update={"acl": ["alice"]}))

    result = chunk_document(doc_model_id, ctx=ctx)
    chunk_set = ctx.store.read_model("test-kb", "chunks", result.output_id, ChunkSet)
    assert [c.position for c in chunk_set.chunks] == [0, 1]
    assert [c.section_path for c in chunk_set.chunks] == [["Refunds"], ["Shipping"]]
    assert all(c.doc_id == raw_id and c.acl == ["alice"] for c in chunk_set.chunks)
    assert chunk_set.source_uri == URI
    assert len({c.chunk_id for c in chunk_set.chunks}) == 2
    assert chunk_document(doc_model_id, ctx=ctx).output_id == result.output_id


def test_new_chunk_settings_rechunk_but_reuse_the_parse(ctx):
    raw_id = read_source(ref(), ctx=ctx).output_id
    doc_model_id = understand_document(raw_id, ctx=ctx).output_id
    first = chunk_document(doc_model_id, ctx=ctx)

    ctx.kb = make_kb({URI: POLICY}, chunk={"strategy": "fixed", "max_tokens": 5, "overlap_tokens": 1})
    assert understand_document(raw_id, ctx=ctx).status == "skipped"
    second = chunk_document(doc_model_id, ctx=ctx)
    assert second.status == "succeeded"
    assert second.output_id != first.output_id


def test_failures_are_logged_and_raised(ctx):
    bad_uri = "mem://docs/bad.md"
    ctx.kb = make_kb({bad_uri: "# Bad"}, parse={"parser": "picky"})
    raw_id = read_source(SourceRef(uri=bad_uri), ctx=ctx).output_id
    with pytest.raises(JobFailed) as err:
        understand_document(raw_id, ctx=ctx)
    assert err.value.stage == "understand_document"
    record = ctx.joblog.get(err.value.job_id)
    assert record.status == "failed"
    assert "cannot parse this one" in record.error
