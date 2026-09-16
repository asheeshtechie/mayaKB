import pytest

from mayakb.ingest.jobs import IngestContext
from mayakb.ingest.runner import ingest
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.joblog import InMemoryJobLog
from mayakb.storage.search.memory import InMemorySearchBackend
from tests.fakes import OverlapReranker, make_kb
from tests.retrieval.helpers import DOCS


@pytest.fixture
def corpus(tmp_path):
    """DOCS ingested into an in-memory backend. Returns (kb config, backend)."""
    kb = make_kb(DOCS, retrieval={"beam": {"min_beam": 1, "max_beam": 3}, "reranker": {"reranker": "overlap"}})
    backend = InMemorySearchBackend()
    ctx = IngestContext(kb=kb, store=FileArtifactStore(tmp_path), joblog=InMemoryJobLog(), backend=backend)
    assert ingest(ctx).counts()["failed"] == 0
    OverlapReranker.calls.clear()
    return kb, backend
