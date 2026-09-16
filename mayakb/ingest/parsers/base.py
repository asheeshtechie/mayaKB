"""Parser interface and helpers shared by every parser."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from mayakb.core.contracts import DocModel, Element, ElementType, RawDocument


class UnsupportedDocument(ValueError):
    """The parser cannot read this kind of file."""


class Parser(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @abstractmethod
    def parse(self, raw: RawDocument, data: bytes) -> DocModel:
        """Turn raw bytes into the common DocModel. doc_id must be raw.id."""


class SectionTracker:
    """The current heading path while walking a document top to bottom."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def heading(self, level: int, text: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    @property
    def path(self) -> list[str]:
        return [text for _, text in self._stack]


class ElementBuilder:
    """Collects elements with sequential ids and the section path they belong to."""

    def __init__(self) -> None:
        self.sections = SectionTracker()
        self.elements: list[Element] = []

    def add(
        self,
        type: ElementType,
        text: str,
        *,
        page: int | None = None,
        level: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        text = text.strip("\n") if type == "code" else text.strip()
        if not text.strip():
            return
        if type == "heading":
            level = level or 1
            self.sections.heading(level, text)
        self.elements.append(
            Element(
                id=f"e{len(self.elements)}",
                type=type,
                text=text,
                page=page,
                section_path=self.sections.path,
                level=level if type == "heading" else None,
                bbox=bbox,
            )
        )


def first_heading(elements: list[Element]) -> str | None:
    return next((e.text for e in elements if e.type == "heading"), None)


def decode_text(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")
