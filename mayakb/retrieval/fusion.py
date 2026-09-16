"""Merge dense and keyword hit lists into one ranked candidate list."""

from __future__ import annotations

from dataclasses import dataclass

from mayakb.core.config import RetrievalConfig
from mayakb.storage.search.base import Hit


@dataclass
class Candidate:
    hit: Hit
    fused_score: float
    dense_rank: int | None = None  # 1-based
    dense_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    kept_by: str | None = None  # why the beam kept it

    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id


def _collect(dense: list[Hit], keyword: list[Hit]) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    for rank, hit in enumerate(dense, start=1):
        candidates[hit.chunk_id] = Candidate(hit=hit, fused_score=0.0, dense_rank=rank, dense_score=hit.score)
    for rank, hit in enumerate(keyword, start=1):
        candidate = candidates.setdefault(hit.chunk_id, Candidate(hit=hit, fused_score=0.0))
        candidate.keyword_rank = rank
        candidate.keyword_score = hit.score
    return candidates


def _ordered(candidates: dict[str, Candidate]) -> list[Candidate]:
    def best_rank(c: Candidate) -> int:
        return min(r for r in (c.dense_rank, c.keyword_rank) if r is not None)

    return sorted(candidates.values(), key=lambda c: (-c.fused_score, best_rank(c), c.chunk_id))


def rrf(dense: list[Hit], keyword: list[Hit], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the lists a chunk appears in."""
    candidates = _collect(dense, keyword)
    for c in candidates.values():
        c.fused_score = sum(1.0 / (k + r) for r in (c.dense_rank, c.keyword_rank) if r is not None)
    return _ordered(candidates)


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    return {key: (s - low) / (high - low) if high > low else 1.0 for key, s in scores.items()}


def weighted(dense: list[Hit], keyword: list[Hit], dense_weight: float = 0.6) -> list[Candidate]:
    """Weighted sum of min-max normalized scores (a chunk missing from a list gets 0 there)."""
    candidates = _collect(dense, keyword)
    dense_norm = _min_max({h.chunk_id: h.score for h in dense})
    keyword_norm = _min_max({h.chunk_id: h.score for h in keyword})
    for chunk_id, c in candidates.items():
        dense_part = dense_weight * dense_norm.get(chunk_id, 0.0)
        c.fused_score = dense_part + (1 - dense_weight) * keyword_norm.get(chunk_id, 0.0)
    return _ordered(candidates)


def fuse(dense: list[Hit], keyword: list[Hit], config: RetrievalConfig) -> list[Candidate]:
    if config.fusion == "weighted":
        return weighted(dense, keyword, config.dense_weight)
    return rrf(dense, keyword, config.rrf_k)
