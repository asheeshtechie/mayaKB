"""Cross-encoder reranker via sentence-transformers. Needs the `rerank` extra; runs on CPU by default."""

from __future__ import annotations

from typing import Any

from mayakb.retrieval.rerankers.base import Reranker


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"
    version = "1"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str | None = None,
        encoder: Any = None,
    ):
        if encoder is None:
            from sentence_transformers import CrossEncoder

            encoder = CrossEncoder(model, device=device)
        self.encoder = encoder
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        import torch

        # Always apply a sigmoid: some models (e.g. ms-marco) are configured to return raw logits.
        scores = self.encoder.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
