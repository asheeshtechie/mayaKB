"""Postgres-only behaviour: stemming, one table per embedder, iterative scans."""

import warnings

import psycopg
import pytest

from mayakb.storage.db import MIN_PGVECTOR, pgvector_version
from mayakb.storage.search.base import Filters, IndexRow
from mayakb.storage.search.postgres import PostgresSearchBackend

pytestmark = pytest.mark.postgres


@pytest.fixture
def backend(pg_pool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PostgresSearchBackend(pg_pool)


def test_keyword_search_uses_stemming(backend):
    index = backend.ensure_index("kb1", "hash", "hash-3", 3)
    backend.upsert([IndexRow(chunk_id="c1", doc_id="d1", position=0, text="Refunds are issued quickly",
                             embedding=[1, 0, 0], source_uri="s3://x")], index)
    assert [h.chunk_id for h in backend.keyword_search("refund", 5, Filters(), index)] == ["c1"]


def test_each_embedder_gets_its_own_table(backend, pg_url):
    a = backend.ensure_index("kb1", "ollama", "nomic-embed-text", 768)
    b = backend.ensure_index("kb1", "ollama", "mxbai-embed-large", 1024)
    assert a.table_name != b.table_name
    with psycopg.connect(pg_url) as conn:
        tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert {a.table_name, b.table_name} <= tables


def test_rejects_dimensions_hnsw_cannot_index(backend):
    with pytest.raises(ValueError):
        backend.ensure_index("kb1", "x", "y", 4096)


def test_iterative_scan_matches_pgvector_version(backend, pg_pool):
    with pg_pool.connection() as conn:
        version = pgvector_version(conn)
    assert backend.iterative_scan == (version >= MIN_PGVECTOR)
