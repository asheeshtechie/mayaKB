import pytest

from mayakb.core import registry
from mayakb.core.contracts import DocModel, Element
from mayakb.ingest.chunking.fixed import FixedChunker


def ten_words(tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(10))


def doc(*elements: Element) -> DocModel:
    return DocModel(doc_id="d1", parser="local", source_uri="file:///x", elements=list(elements))


DOC = doc(
    Element(id="e0", type="paragraph", text=ten_words("a"), page=1, section_path=["A"]),
    Element(id="e1", type="paragraph", text=ten_words("b"), page=1, section_path=["A"]),
    Element(id="e2", type="paragraph", text=ten_words("c"), page=2, section_path=["B"]),
)


def test_windows_with_overlap():
    pieces = FixedChunker(max_tokens=12, overlap_tokens=2).chunk(DOC)
    assert len(pieces) == 3
    first, second, third = pieces
    assert first.text.startswith("a0 ") and first.text.endswith("b1")
    assert second.text.startswith("b0 ")  # 2 words of overlap with the first window
    assert first.element_ids == ["e0", "e1"]
    assert "\n\n" in first.text  # element boundary kept
    assert (third.page_start, third.page_end) == (2, 2)
    assert (second.page_start, second.page_end) == (1, 2)
    assert first.section_path == ["A"]


def test_empty_document_has_no_pieces():
    assert FixedChunker().chunk(doc()) == []


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        FixedChunker(max_tokens=10, overlap_tokens=10)


def test_registered_as_fixed():
    assert registry.get("chunker", "fixed") is FixedChunker
