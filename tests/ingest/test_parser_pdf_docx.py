import io

import docx
import pymupdf

from mayakb.core.contracts import RawDocument
from mayakb.ingest.connectors.base import DOCX_MIME
from mayakb.ingest.parsers.docx import table_to_markdown
from mayakb.ingest.parsers.local import LocalParser


def make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Refund Policy", fontsize=22)
    y = 120
    for i in range(4):
        page.insert_text((72, y), f"Body paragraph {i} explains refunds and returns.", fontsize=11)
        y += 40
    page.insert_text((72, y), "Shipping", fontsize=16)
    page.insert_text((72, y + 40), "Orders ship within two days.", fontsize=11)
    doc.new_page().insert_text((72, 72), "Second page body text about warranty terms.", fontsize=11)
    doc.set_metadata({"title": "Policies 2026"})
    return doc.tobytes()


def make_docx() -> bytes:
    document = docx.Document()
    document.add_heading("Guide", level=1)
    document.add_paragraph("Intro text.")
    document.add_heading("Install", level=2)
    document.add_paragraph("Run the installer", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    for (r, c), text in {(0, 0): "Plan", (0, 1): "Days", (1, 0): "Basic", (1, 1): "30"}.items():
        table.cell(r, c).text = text
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def raw(mime: str) -> RawDocument:
    return RawDocument(id="d1", source_uri="file:///x", connector="local", mime=mime)


def test_pdf_headings_paragraphs_and_pages():
    doc = LocalParser().parse(raw("application/pdf"), make_pdf())
    assert doc.title == "Policies 2026"
    headings = [(e.text, e.level) for e in doc.elements if e.type == "heading"]
    assert headings == [("Refund Policy", 1), ("Shipping", 2)]
    shipping_body = next(e for e in doc.elements if e.text.startswith("Orders ship"))
    assert shipping_body.type == "paragraph"
    assert shipping_body.section_path == ["Refund Policy", "Shipping"]
    assert shipping_body.page == 1
    warranty = next(e for e in doc.elements if "warranty" in e.text)
    assert warranty.page == 2
    assert warranty.bbox is not None


def test_docx_headings_lists_and_tables():
    doc = LocalParser().parse(raw(DOCX_MIME), make_docx())
    got = [(e.type, e.text, e.section_path) for e in doc.elements]
    assert got == [
        ("heading", "Guide", ["Guide"]),
        ("paragraph", "Intro text.", ["Guide"]),
        ("heading", "Install", ["Guide", "Install"]),
        ("list", "Run the installer", ["Guide", "Install"]),
        ("table", "| Plan | Days |\n| --- | --- |\n| Basic | 30 |", ["Guide", "Install"]),
    ]
    assert doc.title == "Guide"


def test_table_to_markdown_pads_and_escapes():
    assert table_to_markdown([["a|b", "c"], ["1"], ["", ""]]) == "| a\\|b | c |\n| --- | --- |\n| 1 |  |"
    assert table_to_markdown([]) == ""
