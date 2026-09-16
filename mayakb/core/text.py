"""Text helpers. A token is approximated as one whitespace-separated word."""

from __future__ import annotations

import re

_PIECE = re.compile(r"\S+\s*")
_WORD = re.compile(r"\w+")


def count_tokens(text: str) -> int:
    return len(text.split())


def word_pieces(text: str) -> list[str]:
    """Words with their trailing whitespace, so "".join(pieces) == text.lstrip()."""
    return _PIECE.findall(text)


def words(text: str) -> list[str]:
    """Lowercase word characters only, for simple term matching."""
    return _WORD.findall(text.lower())


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) windows over n items, each `size` long, overlapping by `overlap`."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        out.append((start, end))
        if end == n:
            break
        start += size - overlap
    return out


def normalize_entity(text: str) -> str:
    """Lowercase with single spaces, so "Acme  Corp" and "acme corp" match."""
    return " ".join(text.lower().split())
