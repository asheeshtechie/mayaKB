"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class EmbeddingError(RuntimeError):
    pass


class Embedder(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    model: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """A vector for a search query (some models use a different prefix than for documents)."""
