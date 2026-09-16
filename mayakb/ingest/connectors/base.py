"""Connector interface: list documents in a source and fetch their bytes."""

from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from mayakb.core.contracts import RawDocument, SourceRef

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

DEFAULT_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def guess_mime(name: str) -> str:
    lower = name.lower()
    for suffix, mime in _MIME_BY_SUFFIX.items():
        if lower.endswith(suffix):
            return mime
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


class Connector(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @abstractmethod
    def list(self) -> Iterator[SourceRef]:
        """Every document in the source, with an etag when the source provides one."""

    @abstractmethod
    def fetch(self, ref: SourceRef) -> tuple[RawDocument, bytes]:
        """The document's bytes and its metadata. RawDocument.id must be content_id(bytes)."""
