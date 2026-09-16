"""Postgres SearchBackend: pgvector HNSW for dense search, full-text search for keywords."""

from __future__ import annotations

import re
import warnings

import numpy as np
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from mayakb.storage.db import MIN_PGVECTOR, DatabaseUnavailable, pgvector_version
from mayakb.storage.search.base import Filters, Hit, IndexInfo, IndexRow, index_id_for, index_table_name

MAX_HNSW_DIM = 2000  # pgvector's HNSW limit for the vector type
_HIT_COLUMNS = (
    "chunk_id", "doc_id", "position", "text", "source_uri", "title",
    "page_start", "page_end", "section_path", "metadata",
)
_INDEX_COLUMNS = "index_id, kb, embedder, model, dim, table_name, text_search_language"


def _columns() -> sql.Composable:
    return sql.SQL(", ").join(sql.Identifier(c) for c in _HIT_COLUMNS)


def _like_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _where(filters: Filters) -> tuple[sql.Composable, list[object]]:
    clauses: list[sql.Composable] = [sql.SQL("TRUE")]
    params: list[object] = []
    if filters.source_prefix:
        clauses.append(sql.SQL("source_uri LIKE %s"))
        params.append(_like_prefix(filters.source_prefix))
    if filters.entities:
        clauses.append(sql.SQL("entities && %s::text[]"))
        params.append(list(filters.entities))
    if filters.principals is not None:
        clauses.append(sql.SQL("(acl IS NULL OR acl && %s::text[])"))
        params.append(list(filters.principals))
    return sql.SQL(" AND ").join(clauses), params


class PostgresSearchBackend:
    name = "postgres"
    version = "1"

    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        with pool.connection() as conn:
            found = pgvector_version(conn)
        if found is None:
            raise DatabaseUnavailable("the pgvector extension is missing; run `mayakb db init`")
        self.iterative_scan = found >= MIN_PGVECTOR
        if not self.iterative_scan:
            warnings.warn(
                f"pgvector {'.'.join(map(str, found))} is older than 0.8: filtered vector searches "
                "may return fewer hits than requested",
                stacklevel=2,
            )

    # -- index management ---------------------------------------------------------------------

    def _index_rows(self, where: str, params: tuple) -> list[IndexInfo]:
        with self.pool.connection() as conn:
            rows = conn.cursor(row_factory=dict_row).execute(
                f"SELECT {_INDEX_COLUMNS} FROM indexes WHERE {where} ORDER BY created_at", params
            ).fetchall()
        return [IndexInfo(**row) for row in rows]

    def find_index(self, kb, embedder, model) -> IndexInfo | None:
        rows = self._index_rows("kb = %s AND embedder = %s AND model = %s", (kb, embedder, model))
        return rows[-1] if rows else None

    def list_indexes(self, kb) -> list[IndexInfo]:
        return self._index_rows("kb = %s", (kb,))

    def ensure_index(self, kb, embedder, model, dim, text_search_language="english") -> IndexInfo:
        if not 0 < dim <= MAX_HNSW_DIM:
            raise ValueError(f"embedding dimension {dim} is outside 1..{MAX_HNSW_DIM}")
        if not re.fullmatch(r"[a-z_]+", text_search_language):
            raise ValueError(f"invalid text search language {text_search_language!r}")
        index_id = index_id_for(kb, embedder, model, dim)
        table = index_table_name(kb, embedder, model, dim)
        t = sql.Identifier(table)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (table,))
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t} (
                        chunk_id     text PRIMARY KEY,
                        doc_id       text NOT NULL,
                        position     integer NOT NULL,
                        text         text NOT NULL,
                        embedding    vector({dim}) NOT NULL,
                        tsv          tsvector GENERATED ALWAYS AS (to_tsvector({lang}, text)) STORED,
                        source_uri   text NOT NULL,
                        title        text,
                        page_start   integer,
                        page_end     integer,
                        section_path text[] NOT NULL DEFAULT '{{}}',
                        entities     text[] NOT NULL DEFAULT '{{}}',
                        acl          text[],
                        metadata     jsonb NOT NULL DEFAULT '{{}}'
                    )
                    """
                ).format(t=t, dim=sql.Literal(dim), lang=sql.Literal(text_search_language))
            )
            for suffix, method, expression in (
                ("hnsw", "hnsw", "embedding vector_cosine_ops"),
                ("tsv", "gin", "tsv"),
                ("entities", "gin", "entities"),
                ("acl", "gin", "acl"),
                ("doc", "btree", "doc_id"),
                ("uri", "btree", "source_uri"),
            ):
                conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {t} USING {method} ({expression})").format(
                        name=sql.Identifier(f"{table}_{suffix}"),
                        t=t,
                        method=sql.SQL(method),
                        expression=sql.SQL(expression),
                    )
                )
            conn.execute(
                f"INSERT INTO indexes ({_INDEX_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (index_id) DO NOTHING",
                (index_id, kb, embedder, model, dim, table, text_search_language),
            )
        return self._index_rows("index_id = %s", (index_id,))[0]

    # -- writes -------------------------------------------------------------------------------

    def upsert(self, rows: list[IndexRow], index: IndexInfo) -> None:
        if not rows:
            return
        t = sql.Identifier(index.table_name)
        doc_ids = sorted({r.doc_id for r in rows})
        uris = sorted({r.source_uri for r in rows})
        chunk_ids = [r.chunk_id for r in rows]
        insert = sql.SQL(
            """
            INSERT INTO {t} (chunk_id, doc_id, position, text, embedding, source_uri, title,
                             page_start, page_end, section_path, entities, acl, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id, position = EXCLUDED.position, text = EXCLUDED.text,
                embedding = EXCLUDED.embedding, source_uri = EXCLUDED.source_uri, title = EXCLUDED.title,
                page_start = EXCLUDED.page_start, page_end = EXCLUDED.page_end,
                section_path = EXCLUDED.section_path, entities = EXCLUDED.entities,
                acl = EXCLUDED.acl, metadata = EXCLUDED.metadata
            """
        ).format(t=t)
        values = [
            (
                r.chunk_id, r.doc_id, r.position, r.text, np.asarray(r.embedding, dtype=np.float32),
                r.source_uri, r.title, r.page_start, r.page_end, list(r.section_path), list(r.entities),
                r.acl, Jsonb(r.metadata),
            )
            for r in rows
        ]
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                sql.SQL(
                    "DELETE FROM {t} WHERE (doc_id = ANY(%s) AND NOT chunk_id = ANY(%s))"
                    " OR (source_uri = ANY(%s) AND NOT doc_id = ANY(%s))"
                ).format(t=t),
                (doc_ids, chunk_ids, uris, doc_ids),
            )
            with conn.cursor() as cur:
                cur.executemany(insert, values)

    # -- searches -----------------------------------------------------------------------------

    def dense_search(self, vector, k, filters, index) -> list[Hit]:
        where, params = _where(filters)
        query = sql.SQL(
            "SELECT {columns}, 1 - (embedding <=> %s) AS score FROM {t} WHERE {where} "
            "ORDER BY embedding <=> %s LIMIT %s"
        ).format(columns=_columns(), t=sql.Identifier(index.table_name), where=where)
        embedding = np.asarray(vector, dtype=np.float32)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(min(max(k, 40), 1000))))
            if self.iterative_scan:
                conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = conn.cursor(row_factory=dict_row).execute(query, [embedding, *params, embedding, k]).fetchall()
        return [Hit(**row) for row in rows]

    def keyword_search(self, text, k, filters, index) -> list[Hit]:
        where, params = _where(filters)
        query = sql.SQL(
            "SELECT {columns}, ts_rank_cd(tsv, q) AS score "
            "FROM {t}, websearch_to_tsquery({lang}::regconfig, %s) AS q "
            "WHERE tsv @@ q AND {where} ORDER BY score DESC, chunk_id LIMIT %s"
        ).format(
            columns=_columns(),
            t=sql.Identifier(index.table_name),
            lang=sql.Literal(index.text_search_language),
            where=where,
        )
        with self.pool.connection() as conn:
            rows = conn.cursor(row_factory=dict_row).execute(query, [text, *params, k]).fetchall()
        return [Hit(**row) for row in rows]
