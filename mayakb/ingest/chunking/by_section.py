"""Chunks follow sections: split at headings, keep tables whole, split only what is too big."""

from __future__ import annotations

from mayakb.core.contracts import ChunkPiece, DocModel, Element
from mayakb.core.text import count_tokens
from mayakb.ingest.chunking.base import (
    ChunkingStrategy,
    check_sizes,
    element_words,
    piece_from,
    window_pieces,
)


def split_sections(elements: list[Element]) -> list[list[Element]]:
    """A new section starts at a heading that follows content. Consecutive headings stay together."""
    sections: list[list[Element]] = []
    current: list[Element] = []
    for element in elements:
        if element.type == "heading" and any(e.type != "heading" for e in current):
            sections.append(current)
            current = []
        current.append(element)
    if current:
        sections.append(current)
    return sections


class BySectionChunker(ChunkingStrategy):
    name = "by_section"
    version = "1"

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        check_sizes(max_tokens, overlap_tokens)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: DocModel) -> list[ChunkPiece]:
        pieces: list[ChunkPiece] = []
        for section in split_sections(doc.elements):
            pieces.extend(self._chunk_section(section))
        return pieces

    def _chunk_section(self, elements: list[Element]) -> list[ChunkPiece]:
        section_path = elements[-1].section_path  # the deepest heading path in this section
        pieces: list[ChunkPiece] = []
        batch: list[Element] = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if batch:
                pieces.append(piece_from("\n\n".join(e.text for e in batch), batch, section_path))
            batch, batch_tokens = [], 0

        for element in elements:
            tokens = count_tokens(element.text)
            if tokens > self.max_tokens:
                flush()
                if element.type == "table":
                    pieces.extend(self._split_table(element))
                else:
                    pieces.extend(window_pieces(element_words([element]), self.max_tokens, self.overlap_tokens))
                continue
            if batch_tokens + tokens > self.max_tokens:
                flush()
            batch.append(element)
            batch_tokens += tokens
        flush()
        return pieces

    def _split_table(self, table: Element) -> list[ChunkPiece]:
        """Split a markdown table by rows, repeating the header in every piece."""
        lines = table.text.splitlines()
        is_separator = len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ")
        header = lines[:2] if is_separator else lines[:1]
        budget = max(self.max_tokens - count_tokens("\n".join(header)), 1)
        pieces: list[ChunkPiece] = []
        rows: list[str] = []
        used = 0
        for row in lines[len(header):]:
            tokens = count_tokens(row)
            if rows and used + tokens > budget:
                pieces.append(piece_from("\n".join(header + rows), [table]))
                rows, used = [], 0
            rows.append(row)
            used += tokens
        if rows:
            pieces.append(piece_from("\n".join(header + rows), [table]))
        return pieces
