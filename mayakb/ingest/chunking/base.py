"""ChunkingStrategy interface and helpers for building chunk pieces from elements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from mayakb.core.contracts import ChunkPiece, DocModel, Element
from mayakb.core.text import windows, word_pieces


class ChunkingStrategy(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @abstractmethod
    def chunk(self, doc: DocModel) -> list[ChunkPiece]:
        """Split a document into pieces, in reading order."""


def check_sizes(max_tokens: int, overlap_tokens: int) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens")


def piece_from(text: str, elements: list[Element], section_path: list[str] | None = None) -> ChunkPiece:
    """A piece covering `elements`; pages span them, section defaults to the first element's."""
    pages = [e.page for e in elements if e.page is not None]
    return ChunkPiece(
        text=text.strip(),
        element_ids=list(dict.fromkeys(e.id for e in elements)),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_path=list(section_path if section_path is not None else elements[0].section_path),
    )


def element_words(elements: list[Element]) -> list[tuple[str, Element]]:
    """Every word with the element it came from; elements are separated by a blank line."""
    out: list[tuple[str, Element]] = []
    for element in elements:
        pieces = word_pieces(element.text)
        if not pieces:
            continue
        pieces[-1] = pieces[-1].rstrip() + "\n\n"
        out.extend((piece, element) for piece in pieces)
    return out


def window_pieces(words: list[tuple[str, Element]], max_tokens: int, overlap_tokens: int) -> list[ChunkPiece]:
    pieces = []
    for start, end in windows(len(words), max_tokens, overlap_tokens):
        span = words[start:end]
        pieces.append(piece_from("".join(word for word, _ in span), [element for _, element in span]))
    return pieces
