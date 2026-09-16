from mayakb.core import registry
from mayakb.core.contracts import DocModel, Element
from mayakb.core.text import count_tokens
from mayakb.ingest.chunking.by_section import BySectionChunker, split_sections
from mayakb.ingest.parsers.base import ElementBuilder
from mayakb.ingest.parsers.markdown import parse_markdown_into


def doc_from_markdown(text: str) -> DocModel:
    builder = ElementBuilder()
    parse_markdown_into(builder, text)
    return DocModel(doc_id="d1", parser="local", source_uri="file:///x", elements=builder.elements)


def test_one_chunk_per_small_section():
    doc = doc_from_markdown("# Refunds\n\nRefunds take 30 days.\n\n# Shipping\n\nOrders ship in two days.")
    pieces = BySectionChunker(max_tokens=50, overlap_tokens=5).chunk(doc)
    assert [p.text for p in pieces] == [
        "Refunds\n\nRefunds take 30 days.",
        "Shipping\n\nOrders ship in two days.",
    ]
    assert [p.section_path for p in pieces] == [["Refunds"], ["Shipping"]]
    assert pieces[0].element_ids == ["e0", "e1"]


def test_consecutive_headings_join_the_following_content():
    doc = doc_from_markdown("# Guide\n\n## Install\n\nRun the installer.")
    assert len(split_sections(doc.elements)) == 1
    [piece] = BySectionChunker(max_tokens=50, overlap_tokens=5).chunk(doc)
    assert piece.section_path == ["Guide", "Install"]
    assert piece.text.startswith("Guide\n\nInstall")


def test_large_section_is_split_between_elements():
    paragraphs = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(8)) for p in range(5))
    doc = doc_from_markdown(f"# Big\n\n{paragraphs}")
    pieces = BySectionChunker(max_tokens=20, overlap_tokens=2).chunk(doc)
    assert len(pieces) > 1
    assert all(count_tokens(p.text) <= 20 for p in pieces)
    assert all(p.section_path == ["Big"] for p in pieces)


def test_oversized_paragraph_uses_overlapping_windows():
    long_text = " ".join(f"x{i}" for i in range(30))
    doc = DocModel(doc_id="d", parser="local", source_uri="file:///x", elements=[
        Element(id="e0", type="paragraph", text=long_text, section_path=["S"]),
    ])
    pieces = BySectionChunker(max_tokens=12, overlap_tokens=2).chunk(doc)
    assert len(pieces) == 3
    assert pieces[1].text.split()[0] == "x10"


def test_oversized_table_repeats_header_in_every_piece():
    rows = "\n".join(f"| item{i} | {i} |" for i in range(12))
    table = f"| Name | Qty |\n| --- | --- |\n{rows}"
    doc = DocModel(doc_id="d", parser="local", source_uri="file:///x", elements=[
        Element(id="e0", type="table", text=table, section_path=["Stock"]),
    ])
    pieces = BySectionChunker(max_tokens=30, overlap_tokens=0).chunk(doc)
    assert len(pieces) > 1
    assert all(p.text.startswith("| Name | Qty |\n| --- | --- |\n") for p in pieces)
    assert sum(p.text.count("| item") for p in pieces) == 12


def test_registered_as_by_section():
    assert registry.get("chunker", "by_section") is BySectionChunker
