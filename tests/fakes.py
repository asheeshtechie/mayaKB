"""Test implementations registered under fake names. Importing this module registers them."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from mayakb.core.config import KBConfig
from mayakb.core.contracts import Chunk, DocModel, Entity, RawDocument, SourceRef
from mayakb.core.ids import content_id
from mayakb.core.registry import register
from mayakb.core.text import normalize_entity, words
from mayakb.ingest.connectors.base import Connector, guess_mime
from mayakb.ingest.embedders.base import Embedder
from mayakb.ingest.ner.base import EntityExtractor
from mayakb.ingest.parsers.base import Parser
from mayakb.ingest.parsers.local import LocalParser


@register("connector", "fake")
class FakeConnector(Connector):
    """Documents given inline in the KB config: {uri: text}."""

    version = "1"

    def __init__(self, docs: dict[str, str]):
        self.docs = docs

    def list(self):
        for uri in sorted(self.docs):
            yield SourceRef(uri=uri, etag=content_id(self.docs[uri].encode())[:12])

    def fetch(self, ref):
        data = self.docs[ref.uri].encode()
        raw = RawDocument(
            id=content_id(data), source_uri=ref.uri, connector=self.name, mime=guess_mime(ref.uri),
            etag=ref.etag, metadata={"filename": ref.uri.rsplit("/", 1)[-1]},
        )
        return raw, data


@register("parser", "picky")
class PickyParser(Parser):
    """Fails for any source whose uri contains 'bad'; otherwise the local parser."""

    version = "1"

    def parse(self, raw: RawDocument, data: bytes) -> DocModel:
        if "bad" in raw.source_uri:
            raise ValueError("cannot parse this one")
        return LocalParser().parse(raw, data)


@register("ner", "capitalized")
class CapitalizedWords(EntityExtractor):
    version = "1"

    def extract(self, chunks: list[Chunk]) -> list[Entity]:
        return [
            Entity(chunk_id=c.chunk_id, text=m.group(0), label="WORD", normalized=normalize_entity(m.group(0)),
                   start=m.start(), end=m.end())
            for c in chunks
            for m in re.finditer(r"\b[A-Z][a-z]+\b", c.text)
        ]


@register("embedder", "hash")
class HashEmbedder(Embedder):
    """Bag of hashed words: texts sharing words get similar vectors. Deterministic, no model."""

    version = "1"

    def __init__(self, dim: int = 64, model: str = "hash-64"):
        self.dim = dim
        self.model = model

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim)
        for word in words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_kb(docs: dict[str, str] | None = None, **sections) -> KBConfig:
    data = {
        "kb": "test-kb",
        "source": {"connector": "fake", "docs": docs or {}},
        "parse": {"parser": "local"},
        "chunk": {"strategy": "by_section", "max_tokens": 60, "overlap_tokens": 5},
        "ner": {"extractor": "capitalized"},
        "embed": {"embedder": "hash", "dim": 64},
        "index": {"backend": "memory"},
    }
    data.update(sections)
    return KBConfig.model_validate(data)


