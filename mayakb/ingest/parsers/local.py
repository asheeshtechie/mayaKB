"""The no-network parser: Markdown and plain text."""

from __future__ import annotations

from mayakb.core.contracts import DocModel, RawDocument
from mayakb.ingest.parsers.base import (
    ElementBuilder,
    Parser,
    UnsupportedDocument,
    decode_text,
    first_heading,
)
from mayakb.ingest.parsers.markdown import parse_markdown_into


class LocalParser(Parser):
    name = "local"
    version = "1"

    def parse(self, raw: RawDocument, data: bytes) -> DocModel:
        builder = ElementBuilder()
        if raw.mime == "text/markdown":
            parse_markdown_into(builder, decode_text(data))
        elif raw.mime.startswith("text/"):
            parse_markdown_into(builder, decode_text(data), headings=False)
        else:
            raise UnsupportedDocument(f"local parser cannot read {raw.mime} ({raw.source_uri})")
        return DocModel(
            doc_id=raw.id,
            parser=self.name,
            source_uri=raw.source_uri,
            title=first_heading(builder.elements) or raw.metadata.get("filename"),
            acl=raw.acl,
            metadata={"mime": raw.mime},
            elements=builder.elements,
        )
