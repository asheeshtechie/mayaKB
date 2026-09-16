"""Data passed between pipeline stages. Every job reads and writes these models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ElementType = Literal["heading", "paragraph", "list", "table", "figure", "code"]


class SourceRef(BaseModel):
    """A document a connector can fetch."""

    uri: str
    etag: str | None = None
    modified_at: datetime | None = None
    size: int | None = None


class RawDocument(BaseModel):
    """Metadata for fetched bytes. The bytes are stored next to it."""

    id: str  # sha256 of the bytes
    source_uri: str
    connector: str
    mime: str
    etag: str | None = None
    modified_at: datetime | None = None
    acl: list[str] | None = None  # reserved: principals allowed to read; None = unrestricted
    metadata: dict[str, Any] = Field(default_factory=dict)


class Element(BaseModel):
    id: str
    type: ElementType
    text: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None


class DocModel(BaseModel):
    """The common output of every parser."""

    doc_id: str  # RawDocument.id, so it stays the same across parsers and chunkers
    parser: str
    source_uri: str
    title: str | None = None
    acl: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    elements: list[Element] = Field(default_factory=list)


class ChunkPiece(BaseModel):
    """What a chunking strategy returns. The chunk job adds ids, position and acl."""

    text: str
    element_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(ChunkPiece):
    chunk_id: str
    doc_id: str
    position: int
    acl: list[str] | None = None


class ChunkSet(BaseModel):
    id: str
    doc_id: str
    source_uri: str
    title: str | None = None
    strategy: str
    chunks: list[Chunk] = Field(default_factory=list)


class Entity(BaseModel):
    chunk_id: str
    text: str
    label: str
    normalized: str
    start: int
    end: int
    score: float | None = None


class EntitySet(BaseModel):
    id: str
    chunk_set_id: str
    extractor: str
    entities: list[Entity] = Field(default_factory=list)


class VectorSet(BaseModel):
    """Metadata for a vectors .npy file: row i belongs to chunk_ids[i]."""

    id: str
    chunk_set_id: str
    embedder: str
    model: str
    dim: int
    chunk_ids: list[str] = Field(default_factory=list)
