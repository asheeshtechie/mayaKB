import pytest

spacy = pytest.importorskip("spacy")

from mayakb.core import registry  # noqa: E402
from mayakb.core.contracts import Chunk  # noqa: E402
from mayakb.ingest.ner.spacy_ner import SpacyExtractor  # noqa: E402


@pytest.fixture
def nlp():
    """A blank pipeline with fixed patterns, so tests need no model download."""
    pipeline = spacy.blank("en")
    ruler = pipeline.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "ORG", "pattern": "Acme Corp"},
        {"label": "GPE", "pattern": "EMEA"},
    ])
    return pipeline


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d1", position=0, text=text)


def test_extracts_entities_with_offsets_and_normalized_text(nlp):
    chunks = [chunk("c1", "Acme Corp grew in EMEA."), chunk("c2", "Nothing here.")]
    entities = SpacyExtractor(nlp=nlp).extract(chunks)
    assert [(e.chunk_id, e.text, e.label, e.normalized) for e in entities] == [
        ("c1", "Acme Corp", "ORG", "acme corp"),
        ("c1", "EMEA", "GPE", "emea"),
    ]
    first = entities[0]
    assert chunks[0].text[first.start:first.end] == "Acme Corp"


def test_label_filter(nlp):
    entities = SpacyExtractor(nlp=nlp, labels=["GPE"]).extract([chunk("c1", "Acme Corp grew in EMEA.")])
    assert [e.label for e in entities] == ["GPE"]


def test_no_chunks(nlp):
    assert SpacyExtractor(nlp=nlp).extract([]) == []


def test_registered_as_spacy():
    assert registry.get("ner", "spacy") is SpacyExtractor
