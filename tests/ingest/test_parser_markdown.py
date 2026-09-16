import pytest

from mayakb.core import registry
from mayakb.core.contracts import RawDocument
from mayakb.ingest.parsers.base import ElementBuilder, SectionTracker, UnsupportedDocument
from mayakb.ingest.parsers.local import LocalParser
from mayakb.ingest.parsers.markdown import parse_markdown_into

MARKDOWN = """# Refund Policy

Refunds are issued within 30 days.
Contact support to start one.

## Eligibility

- Item unused
- Receipt included
  with order number

| Plan | Days |
|------|------|
| Basic | 30 |

```python
def refund():
    return True
```

# Shipping ##
Orders ship in two days.
"""


def raw(mime: str, source: str = "file:///docs/policy.md") -> RawDocument:
    return RawDocument(id="doc1", source_uri=source, connector="local", mime=mime, metadata={"filename": "policy.md"})


def test_section_tracker_pops_to_parent_level():
    tracker = SectionTracker()
    tracker.heading(1, "A")
    tracker.heading(2, "B")
    tracker.heading(3, "C")
    tracker.heading(2, "D")
    assert tracker.path == ["A", "D"]


def test_markdown_elements_and_sections():
    builder = ElementBuilder()
    parse_markdown_into(builder, MARKDOWN)
    got = [(e.type, e.text.splitlines()[0], e.section_path) for e in builder.elements]
    assert got == [
        ("heading", "Refund Policy", ["Refund Policy"]),
        ("paragraph", "Refunds are issued within 30 days.", ["Refund Policy"]),
        ("heading", "Eligibility", ["Refund Policy", "Eligibility"]),
        ("list", "- Item unused", ["Refund Policy", "Eligibility"]),
        ("table", "| Plan | Days |", ["Refund Policy", "Eligibility"]),
        ("code", "def refund():", ["Refund Policy", "Eligibility"]),
        ("heading", "Shipping", ["Shipping"]),
        ("paragraph", "Orders ship in two days.", ["Shipping"]),
    ]
    assert [e.id for e in builder.elements] == [f"e{i}" for i in range(8)]
    assert builder.elements[3].text.endswith("with order number")
    assert builder.elements[5].text == "def refund():\n    return True"


def test_plain_text_has_no_headings():
    builder = ElementBuilder()
    parse_markdown_into(builder, "# not a heading\n\nsecond paragraph", headings=False)
    assert [e.type for e in builder.elements] == ["paragraph", "paragraph"]


def test_local_parser_markdown_doc_model():
    doc = LocalParser().parse(raw("text/markdown"), MARKDOWN.encode())
    assert doc.doc_id == "doc1"
    assert doc.parser == "local"
    assert doc.title == "Refund Policy"
    assert doc.source_uri == "file:///docs/policy.md"
    assert len(doc.elements) == 8


def test_local_parser_title_falls_back_to_filename():
    doc = LocalParser().parse(raw("text/plain"), b"just text")
    assert doc.title == "policy.md"


def test_local_parser_rejects_unknown_types():
    with pytest.raises(UnsupportedDocument):
        LocalParser().parse(raw("image/png"), b"png")


def test_registered_as_local():
    assert registry.get("parser", "local") is LocalParser
