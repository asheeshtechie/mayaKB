"""Reranker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class Reranker(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, between 0 and 1, in the same order as `texts`."""
