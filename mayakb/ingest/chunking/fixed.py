"""Fixed-size windows of tokens with overlap, ignoring document structure."""

from __future__ import annotations

from mayakb.core.contracts import ChunkPiece, DocModel
from mayakb.ingest.chunking.base import ChunkingStrategy, check_sizes, element_words, window_pieces


class FixedChunker(ChunkingStrategy):
    name = "fixed"
    version = "1"

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        check_sizes(max_tokens, overlap_tokens)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: DocModel) -> list[ChunkPiece]:
        return window_pieces(element_words(doc.elements), self.max_tokens, self.overlap_tokens)
