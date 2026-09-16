"""PDF -> elements with PyMuPDF. Headings are detected from font size, tables with find_tables()."""

from __future__ import annotations

from collections import Counter

import pymupdf

from mayakb.ingest.parsers.base import ElementBuilder

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # otherwise PyMuPDF prints an advert to stdout

HEADING_SIZE_RATIO = 1.15  # a block this much larger than body text is a heading
MAX_HEADING_CHARS = 200


def _overlaps(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _is_heading(text: str, size: float, body_size: float) -> bool:
    return body_size > 0 and size >= body_size * HEADING_SIZE_RATIO and len(text) <= MAX_HEADING_CHARS


def parse_pdf_into(builder: ElementBuilder, data: bytes) -> str | None:
    """Add elements for every page. Returns the PDF's title metadata, if set."""
    pages: list[tuple[int, list[tuple[float, str, object, tuple]]]] = []
    chars_by_size: Counter[float] = Counter()
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        title = (doc.metadata or {}).get("title") or None
        for page_number, page in enumerate(doc, start=1):
            items: list[tuple[float, str, object, tuple]] = []
            table_boxes: list[tuple] = []
            try:
                for table in page.find_tables().tables:
                    box = tuple(table.bbox)
                    table_boxes.append(box)
                    items.append((box[1], "table", table.to_markdown(), box))
            except Exception:  # table detection is best effort; fall back to plain text
                items, table_boxes = [], []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                box = tuple(block["bbox"])
                if any(_overlaps(box, t) for t in table_boxes):
                    continue
                spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "\n".join("".join(s["text"] for s in line["spans"]) for line in block["lines"])
                size = round(max(s["size"] for s in spans), 1)
                for span in spans:
                    chars_by_size[round(span["size"], 1)] += len(span["text"])
                items.append((box[1], "text", (text.strip(), size), box))
            items.sort(key=lambda item: item[0])
            pages.append((page_number, items))

    body_size = chars_by_size.most_common(1)[0][0] if chars_by_size else 0.0
    heading_sizes = sorted(
        {
            payload[1]
            for _, items in pages
            for _, kind, payload, _ in items
            if kind == "text" and _is_heading(payload[0], payload[1], body_size)
        },
        reverse=True,
    )
    level_by_size = {size: min(i + 1, 6) for i, size in enumerate(heading_sizes)}

    for page_number, items in pages:
        for _, kind, payload, box in items:
            if kind == "table":
                builder.add("table", payload, page=page_number, bbox=box)
                continue
            text, size = payload
            if _is_heading(text, size, body_size):
                builder.add("heading", " ".join(text.split()), page=page_number, level=level_by_size[size], bbox=box)
            else:
                builder.add("paragraph", text, page=page_number, bbox=box)
    return title
