"""Contract tests every SearchBackend must pass."""

import warnings

import pytest

from mayakb.storage.search.base import Filters, IndexRow, index_table_name
from mayakb.storage.search.memory import InMemorySearchBackend
from mayakb.storage.search.postgres import PostgresSearchBackend


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
def backend(request):
    if request.param == "memory":
        return InMemorySearchBackend()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # older pgvector in local test databases
        return PostgresSearchBackend(request.getfixturevalue("pg_pool"))


def row(chunk_id, text, embedding, doc_id="d1", uri="s3://docs/a.md", position=0, entities=(), acl=None):
    return IndexRow(
        chunk_id=chunk_id, doc_id=doc_id, position=position, text=text, embedding=list(embedding),
        source_uri=uri, title="A", page_start=1, page_end=1, section_path=["A"],
        entities=list(entities), acl=acl, metadata={"k": "v"},
    )


@pytest.fixture
def index(backend):
    return backend.ensure_index("kb1", "hash", "hash-3", 3)


NONE = Filters()


def test_ensure_index_is_idempotent_and_findable(backend, index):
    assert backend.ensure_index("kb1", "hash", "hash-3", 3) == index
    assert backend.find_index("kb1", "hash", "hash-3") == index
    assert backend.find_index("kb1", "hash", "other") is None
    assert [i.index_id for i in backend.list_indexes("kb1")] == [index.index_id]
    assert index.table_name == index_table_name("kb1", "hash", "hash-3", 3)


def test_dense_search_ranks_by_cosine(backend, index):
    backend.upsert([
        row("c1", "refund policy", [1, 0, 0]),
        row("c2", "shipping times", [0, 1, 0], position=1),
        row("c3", "refund shipping", [0.7, 0.7, 0], position=2),
    ], index)
    hits = backend.dense_search([1, 0, 0], 2, NONE, index)
    assert [h.chunk_id for h in hits] == ["c1", "c3"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[0].section_path == ["A"] and hits[0].metadata == {"k": "v"}


def test_keyword_search_returns_only_matches(backend, index):
    backend.upsert([
        row("c1", "refund policy for refund requests", [1, 0, 0]),
        row("c2", "shipping times", [0, 1, 0], position=1),
    ], index)
    hits = backend.keyword_search("refund", 10, NONE, index)
    assert [h.chunk_id for h in hits] == ["c1"]
    assert hits[0].score > 0


def test_filters_apply_inside_the_search(backend, index):
    backend.upsert([
        row("c1", "refund one", [1, 0, 0], doc_id="d1", uri="s3://docs/2026/a.md", entities=["emea"]),
        row("c2", "refund two", [1, 0, 0], doc_id="d2", uri="s3://docs/2025/b.md", acl=["alice"]),
        row("c3", "refund three", [1, 0, 0], doc_id="d3", uri="s3://docs/2025/c.md", acl=["bob"]),
    ], index)

    def ids(filters):
        dense = {h.chunk_id for h in backend.dense_search([1, 0, 0], 10, filters, index)}
        keyword = {h.chunk_id for h in backend.keyword_search("refund", 10, filters, index)}
        assert dense == keyword
        return dense

    assert ids(NONE) == {"c1", "c2", "c3"}
    assert ids(Filters(source_prefix="s3://docs/2025/")) == {"c2", "c3"}
    assert ids(Filters(entities=["emea"])) == {"c1"}
    assert ids(Filters(principals=["alice"])) == {"c1", "c2"}
    assert ids(Filters(principals=[])) == {"c1"}


def test_source_prefix_treats_wildcards_literally(backend, index):
    backend.upsert([
        row("c1", "refund", [1, 0, 0], doc_id="d1", uri="s3://docs/a_b/x.md"),
        row("c2", "refund", [1, 0, 0], doc_id="d2", uri="s3://docs/aXb/x.md"),
    ], index)
    hits = backend.dense_search([1, 0, 0], 10, Filters(source_prefix="s3://docs/a_b/"), index)
    assert [h.chunk_id for h in hits] == ["c1"]


def test_upsert_replaces_stale_chunks_and_old_versions(backend, index):
    backend.upsert([row("c1", "old one", [1, 0, 0]), row("c2", "old two", [1, 0, 0], position=1)], index)
    backend.upsert([row("c1", "new one", [1, 0, 0])], index)
    assert {h.text for h in backend.dense_search([1, 0, 0], 10, NONE, index)} == {"new one"}
    # the file changed: new content id, same uri -> the old version's rows go away
    backend.upsert([row("c9", "version two", [1, 0, 0], doc_id="d2")], index)
    assert {h.chunk_id for h in backend.dense_search([1, 0, 0], 10, NONE, index)} == {"c9"}


def test_filtered_dense_search_returns_k_hits(backend, index):
    rows = [
        row(f"c{i:02d}", f"doc {i}", [1, i / 100, 0], doc_id=f"d{i}", uri=f"s3://docs/{'a' if i % 2 else 'b'}/{i}.md")
        for i in range(60)
    ]
    backend.upsert(rows, index)
    hits = backend.dense_search([1, 0, 0], 20, Filters(source_prefix="s3://docs/a/"), index)
    assert len(hits) == 20
