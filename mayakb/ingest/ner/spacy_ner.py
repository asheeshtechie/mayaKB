"""Named entities with spaCy. Needs the `ner` extra and a downloaded model (e.g. en_core_web_sm)."""

from __future__ import annotations

from typing import Any

from mayakb.core.contracts import Chunk, Entity
from mayakb.core.text import normalize_entity
from mayakb.ingest.ner.base import EntityExtractor


class SpacyExtractor(EntityExtractor):
    name = "spacy"
    version = "1"

    def __init__(
        self,
        model: str = "en_core_web_sm",
        labels: list[str] | None = None,
        batch_size: int = 32,
        nlp: Any = None,
    ):
        if nlp is None:
            import spacy

            nlp = spacy.load(model)
        self.nlp = nlp
        self.labels = set(labels) if labels else None
        self.batch_size = batch_size

    def extract(self, chunks: list[Chunk]) -> list[Entity]:
        entities: list[Entity] = []
        docs = self.nlp.pipe((chunk.text for chunk in chunks), batch_size=self.batch_size)
        for chunk, doc in zip(chunks, docs, strict=True):
            for ent in doc.ents:
                if self.labels is not None and ent.label_ not in self.labels:
                    continue
                entities.append(
                    Entity(
                        chunk_id=chunk.chunk_id,
                        text=ent.text,
                        label=ent.label_,
                        normalized=normalize_entity(ent.text),
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
        return entities
