import pytest

from mayakb.ingest.jobs import IngestContext
from mayakb.ingest.runner import ingest
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.joblog import InMemoryJobLog
from mayakb.storage.search.base import Filters
from mayakb.storage.search.memory import InMemorySearchBackend
from tests.fakes import make_kb

DOCS = {
    "mem://docs/good.md": "# Refunds\n\nRefunds take 30 days.",
    "mem://docs/bad.md": "# Broken\n\nThis one fails to parse.",
}


@pytest.fixture
def ctx(tmp_path):
    return IngestContext(kb=make_kb(DOCS, parse={"parser": "picky"}), store=FileArtifactStore(tmp_path),
                         joblog=InMemoryJobLog(), backend=InMemorySearchBackend())


def test_one_bad_document_does_not_stop_the_batch(ctx):
    report = ingest(ctx)
    by_uri = {d.source_uri: d for d in report.documents}
    assert report.counts() == {"succeeded": 1, "skipped": 0, "failed": 1}
    bad = by_uri["mem://docs/bad.md"]
    assert bad.failed_stage == "understand_document"
    assert "cannot parse" in bad.error
    good = by_uri["mem://docs/good.md"]
    assert [j.stage for j in good.jobs] == [
        "read_source", "understand_document", "chunk_document", "embed_chunks", "extract_entities", "index_chunks",
    ]
    index = ctx.backend.find_index("test-kb", "hash", "hash-64")
    assert ctx.backend.keyword_search("refunds", 5, Filters(), index)


def test_second_run_skips_everything_that_succeeded(ctx):
    ingest(ctx)
    again = ingest(ctx)
    assert again.counts() == {"succeeded": 0, "skipped": 1, "failed": 1}


def test_ner_is_optional(ctx):
    ctx.kb = make_kb({"mem://docs/good.md": DOCS["mem://docs/good.md"]}, ner=None)
    report = ingest(ctx)
    assert report.counts()["succeeded"] == 1
    assert "extract_entities" not in [j.stage for j in report.documents[0].jobs]
