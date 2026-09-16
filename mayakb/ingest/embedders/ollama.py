"""Embeddings from an Ollama server (POST /api/embed)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from mayakb.core.config import Settings
from mayakb.core.retry import with_retries
from mayakb.ingest.embedders.base import Embedder, EmbeddingError

# Models trained with task prefixes: (document prefix, query prefix)
PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def _base_model(model: str) -> str:
    return model.split(":", 1)[0].rsplit("/", 1)[-1]


class OllamaEmbedder(Embedder):
    name = "ollama"
    version = "1"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        default_document, default_query = PREFIXES.get(_base_model(model), ("", ""))
        self.model = model
        self.batch_size = batch_size
        self.document_prefix = default_document if document_prefix is None else document_prefix
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url or Settings.from_env().ollama_url, timeout=timeout, transport=transport
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _post(self, batch: list[str]) -> dict[str, Any]:
        response = self._client.post("/api/embed", json={"model": self.model, "input": batch})
        if response.status_code == 404:
            raise EmbeddingError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")
        response.raise_for_status()
        return response.json()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start:start + self.batch_size]
            body = with_retries(lambda batch=batch: self._post(batch), sleep=self._sleep)
            embeddings = body.get("embeddings") or []
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs (model {self.model})"
                )
            vectors.extend(embeddings)
        return vectors
