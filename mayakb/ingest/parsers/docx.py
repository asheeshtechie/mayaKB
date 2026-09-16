"""DOCX -> elements with python-docx. Headings come from paragraph styles."""

from __future__ import annotations

import io
import re

import docx
from docx.table import Table

from mayakb.ingest.parsers.base import ElementBuilder

_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"

    return "\n".join([line(rows[0]), line(["---"] * width), *(line(row) for row in rows[1:])])


def parse_docx_into(builder: ElementBuilder, data: bytes) -> str | None:
    """Add elements in document order. Returns the core-properties title, if set."""
    document = docx.Document(io.BytesIO(data))
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            builder.add("table", table_to_markdown(rows))
            continue
        style = item.style.name if item.style is not None else ""
        heading = _HEADING_STYLE.match(style)
        if style == "Title":
            builder.add("heading", item.text, level=1)
        elif heading:
            builder.add("heading", item.text, level=int(heading.group(1)))
        elif style.startswith("List"):
            builder.add("list", item.text)
        else:
            builder.add("paragraph", item.text)
    return document.core_properties.title or None
