import pytest
from pydantic import ValidationError

from mayakb.core.contracts import (
    Chunk,
    ChunkPiece,
    DocModel,
    Element,
    RawDocument,
    SourceRef,
    VectorSet,
)


def test_raw_document_is_unrestricted_by_default():
    raw = RawDocument(id="abc", source_uri="file:///a.txt", connector="local", mime="text/plain")
    assert raw.acl is None
    assert raw.metadata == {}


def test_element_type_is_validated():
    with pytest.raises(ValidationError):
        Element(id="e0", type="banner", text="x")


def test_doc_model_round_trips_through_json():
    doc = DocModel(
        doc_id="d1",
        parser="local",
        source_uri="file:///a.md",
        title="A",
        elements=[Element(id="e0", type="heading", text="A", level=1, section_path=["A"])],
    )
    assert DocModel.model_validate_json(doc.model_dump_json()) == doc


def test_chunk_extends_piece_and_round_trips():
    piece = ChunkPiece(text="hello", element_ids=["e0"], page_start=1, page_end=2, section_path=["Intro"])
    chunk = Chunk(**piece.model_dump(), chunk_id="c1", doc_id="d1", position=0, acl=["alice"])
    again = Chunk.model_validate_json(chunk.model_dump_json())
    assert again == chunk
    assert again.section_path == ["Intro"]


def test_source_ref_parses_iso_datetime():
    ref = SourceRef.model_validate({"uri": "s3://b/k", "modified_at": "2026-09-16T10:00:00Z"})
    assert ref.modified_at.year == 2026


def test_vector_set_defaults():
    vs = VectorSet(id="v", chunk_set_id="c", embedder="ollama", model="nomic-embed-text", dim=768)
    assert vs.chunk_ids == []
