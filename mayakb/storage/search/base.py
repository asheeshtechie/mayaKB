"""SearchBackend interface: stores index rows and runs dense and keyword searches with filters inside."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, Field

from mayakb.core.ids import stable_id
from mayakb.core.text import normalize_entity


class Filters(BaseModel):
    source_prefix: str | None = None
    entities: list[str] | None = None
    principals: list[str] | None = None  # reserved ACL filter; None means no ACL filtering

    def normalized(self) -> Filters:
        if not self.entities:
            return self
        return self.model_copy(update={"entities": [normalize_entity(e) for e in self.entities]})


class IndexInfo(BaseModel):
    index_id: str
    kb: str
    embedder: str
    model: str
    dim: int
    table_name: str
    text_search_language: str = "english"


class IndexRow(BaseModel):
    chunk_id: str
    doc_id: str
    position: int
    text: str
    embedding: list[float]
    source_uri: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # normalized entity text
    acl: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    position: int
    text: str
    score: float
    source_uri: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchBackend(Protocol):
    name: ClassVar[str]
    version: ClassVar[str]

    def ensure_index(
        self, kb: str, embedder: str, model: str, dim: int, text_search_language: str = "english"
    ) -> IndexInfo: ...
    def find_index(self, kb: str, embedder: str, model: str) -> IndexInfo | None: ...
    def list_indexes(self, kb: str) -> list[IndexInfo]: ...
    def upsert(self, rows: list[IndexRow], index: IndexInfo) -> None:
        """Replace each document's rows. For every doc_id in `rows`, rows not in `rows` are deleted;
        rows with the same source_uri but a different doc_id (an older version of the file) too."""
    def dense_search(self, vector: list[float], k: int, filters: Filters, index: IndexInfo) -> list[Hit]:
        """Top k by cosine similarity (score = 1 - cosine distance)."""
    def keyword_search(self, text: str, k: int, filters: Filters, index: IndexInfo) -> list[Hit]:
        """Top k chunks matching the query terms. Chunks with no match are not returned."""


def index_id_for(kb: str, embedder: str, model: str, dim: int) -> str:
    return stable_id("index", kb, embedder, model, dim)


def index_table_name(kb: str, embedder: str, model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", kb.lower())[:30]
    return f"idx_{slug}_{stable_id(embedder, model, dim, length=12)}"
