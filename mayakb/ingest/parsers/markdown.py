"""Markdown and plain text -> elements."""

from __future__ import annotations

import re

from mayakb.ingest.parsers.base import ElementBuilder

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _line_kind(line: str) -> str:
    if line.lstrip().startswith("|"):
        return "table"
    if _LIST_ITEM.match(line):
        return "list"
    return "paragraph"


def parse_markdown_into(
    builder: ElementBuilder, text: str, *, page: int | None = None, headings: bool = True
) -> None:
    """Add elements for `text`. With headings=False, '#' lines are ordinary text (plain .txt)."""
    lines = text.splitlines()
    block: list[str] = []
    block_kind: str | None = None

    def flush() -> None:
        nonlocal block, block_kind
        if block and block_kind:
            builder.add(block_kind, "\n".join(block), page=page)
        block, block_kind = [], None

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker = fence.group(1)
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(marker):
                code.append(lines[i])
                i += 1
            builder.add("code", "\n".join(code), page=page)
            i += 1  # closing fence
            continue
        heading = _HEADING.match(line) if headings else None
        if heading:
            flush()
            builder.add("heading", heading.group(2), page=page, level=len(heading.group(1)))
        elif not line.strip():
            flush()
        else:
            kind = _line_kind(line)
            # an indented or wrapped line under a list item stays in the list
            continues_list = block_kind == "list" and kind == "paragraph"
            if block_kind is not None and kind != block_kind and not continues_list:
                flush()
            if block_kind is None:
                block_kind = kind
            block.append(line)
        i += 1
    flush()
