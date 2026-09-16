"""In-memory SearchBackend for tests. Same filter and replace rules as Postgres; no stemming."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from mayakb.core.text import words
from mayakb.storage.search.base import Filters, Hit, IndexInfo, IndexRow, index_id_for, index_table_name


def _hit(row: IndexRow, score: float) -> Hit:
    return Hit(score=score, **row.model_dump(exclude={"embedding", "entities", "acl"}))


class InMemorySearchBackend:
    name = "memory"
    version = "1"

    def __init__(self) -> None:
        self._indexes: dict[str, IndexInfo] = {}
        self._rows: dict[str, dict[str, IndexRow]] = {}
        self._lock = threading.Lock()

    def ensure_index(self, kb, embedder, model, dim, text_search_language="english") -> IndexInfo:
        info = IndexInfo(
            index_id=index_id_for(kb, embedder, model, dim),
            kb=kb,
            embedder=embedder,
            model=model,
            dim=dim,
            table_name=index_table_name(kb, embedder, model, dim),
            text_search_language=text_search_language,
        )
        with self._lock:
            self._rows.setdefault(info.index_id, {})
            return self._indexes.setdefault(info.index_id, info)

    def find_index(self, kb, embedder, model) -> IndexInfo | None:
        matches = [i for i in self._indexes.values() if (i.kb, i.embedder, i.model) == (kb, embedder, model)]
        return matches[-1] if matches else None

    def list_indexes(self, kb) -> list[IndexInfo]:
        return [i for i in self._indexes.values() if i.kb == kb]

    def upsert(self, rows: list[IndexRow], index: IndexInfo) -> None:
        doc_ids = {r.doc_id for r in rows}
        uris = {r.source_uri for r in rows}
        keep = {r.chunk_id for r in rows}
        with self._lock:
            table = self._rows[index.index_id]
            for chunk_id, row in list(table.items()):
                stale_chunk = row.doc_id in doc_ids and chunk_id not in keep
                old_version = row.source_uri in uris and row.doc_id not in doc_ids
                if stale_chunk or old_version:
                    del table[chunk_id]
            for row in rows:
                table[row.chunk_id] = row

    def _filtered(self, filters: Filters, index: IndexInfo) -> Iterator[IndexRow]:
        with self._lock:
            rows = list(self._rows.get(index.index_id, {}).values())
        for row in rows:
            if filters.source_prefix and not row.source_uri.startswith(filters.source_prefix):
                continue
            if filters.entities and not set(filters.entities) & set(row.entities):
                continue
            if filters.principals is not None and row.acl is not None and not set(row.acl) & set(filters.principals):
                continue
            yield row

    def dense_search(self, vector, k, filters, index) -> list[Hit]:
        query = np.asarray(vector, dtype=float)
        query_norm = np.linalg.norm(query)
        scored = []
        for row in self._filtered(filters, index):
            emb = np.asarray(row.embedding, dtype=float)
            denom = query_norm * np.linalg.norm(emb)
            scored.append((float(query @ emb / denom) if denom else 0.0, row))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [_hit(row, score) for score, row in scored[:k]]

    def keyword_search(self, text, k, filters, index) -> list[Hit]:
        terms = set(words(text))
        scored = []
        for row in self._filtered(filters, index):
            row_words = words(row.text)
            score = sum(row_words.count(term) for term in terms) / (len(row_words) or 1)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [_hit(row, score) for score, row in scored[:k]]
