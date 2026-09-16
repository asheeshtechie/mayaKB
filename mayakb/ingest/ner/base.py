"""EntityExtractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from mayakb.core.contracts import Chunk, Entity


class EntityExtractor(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @abstractmethod
    def extract(self, chunks: list[Chunk]) -> list[Entity]:
        """Entities found in each chunk. Offsets are character offsets into chunk.text."""
