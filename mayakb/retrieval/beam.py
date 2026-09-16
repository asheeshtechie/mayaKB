"""The dynamic beam (which fused candidates go to the reranker) and the final cut."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

from mayakb.core.config import BeamConfig, CutConfig
from mayakb.core.text import count_tokens
from mayakb.retrieval.fusion import Candidate


@dataclass
class BeamSelection:
    beam: list[Candidate]  # in fused order
    rest: list[Candidate]  # fused candidates left out, in fused order
    kept_by: dict[str, int] = field(default_factory=dict)


class BeamPolicy(ABC):
    @abstractmethod
    def select(self, candidates: list[Candidate]) -> BeamSelection: ...


def _selection(candidates: list[Candidate], kept: list[Candidate]) -> BeamSelection:
    kept_ids = {c.chunk_id for c in kept}
    beam = [c for c in candidates if c.chunk_id in kept_ids]
    rest = [c for c in candidates if c.chunk_id not in kept_ids]
    return BeamSelection(beam=beam, rest=rest, kept_by=dict(Counter(c.kept_by for c in beam)))


class FixedBeam(BeamPolicy):
    """The baseline: the top `size` fused candidates."""

    def __init__(self, size: int = 30):
        self.size = size

    def select(self, candidates: list[Candidate]) -> BeamSelection:
        kept = candidates[: self.size]
        for c in kept:
            c.kept_by = "fixed"
        return _selection(candidates, kept)


class AdaptiveBeam(BeamPolicy):
    """Keep only candidates the cheap retrievers are confident about.

    A candidate is kept if both retrievers rank it in their top `agree_top`, or its dense score is
    within `dense_rel` of the best dense score, or its keyword score within `keyword_rel` of the
    best keyword score. Then at most `max_beam` are kept, topped up to `min_beam` in fused order.
    """

    def __init__(self, min_beam: int = 5, max_beam: int = 30, agree_top: int = 20,
                 dense_rel: float = 0.9, keyword_rel: float = 0.6):
        self.min_beam = min_beam
        self.max_beam = max_beam
        self.agree_top = agree_top
        self.dense_rel = dense_rel
        self.keyword_rel = keyword_rel

    def _reason(self, c: Candidate, best_dense: float | None, best_keyword: float | None) -> str | None:
        if (c.dense_rank is not None and c.keyword_rank is not None
                and c.dense_rank <= self.agree_top and c.keyword_rank <= self.agree_top):
            return "agreement"
        if best_dense and best_dense > 0 and c.dense_score is not None and c.dense_score >= self.dense_rel * best_dense:
            return "dense"
        if (best_keyword and best_keyword > 0 and c.keyword_score is not None
                and c.keyword_score >= self.keyword_rel * best_keyword):
            return "keyword"
        return None

    def select(self, candidates: list[Candidate]) -> BeamSelection:
        best_dense = max((c.dense_score for c in candidates if c.dense_score is not None), default=None)
        best_keyword = max((c.keyword_score for c in candidates if c.keyword_score is not None), default=None)
        kept: list[Candidate] = []
        for c in candidates:
            c.kept_by = self._reason(c, best_dense, best_keyword)
            if c.kept_by:
                kept.append(c)
        kept = kept[: self.max_beam]
        if len(kept) < self.min_beam:
            kept_ids = {c.chunk_id for c in kept}
            for c in candidates:
                if len(kept) >= self.min_beam:
                    break
                if c.chunk_id not in kept_ids:
                    c.kept_by = "top_up"
                    kept.append(c)
        return _selection(candidates, kept)


def make_beam(config: BeamConfig) -> BeamPolicy:
    if config.policy == "fixed":
        return FixedBeam(config.size)
    return AdaptiveBeam(config.min_beam, config.max_beam, config.agree_top, config.dense_rel, config.keyword_rel)


def cut(
    ranked: list[Candidate], config: CutConfig, token_budget: int, *, reranked: bool
) -> tuple[list[Candidate], bool]:
    """Choose the final chunks. Returns (selected, low_confidence).

    `ranked` is best first. When reranked, final k is the smallest of: items scoring >= min_score,
    items within relative_to_top of the best score, and items fitting the token budget; always at
    least min_k and at most max_k. low_confidence is true when nothing reaches min_score.
    """
    if not ranked:
        return [], False
    fits, used = 0, 0
    for c in ranked:
        used += count_tokens(c.hit.text)
        if used > token_budget:
            break
        fits += 1
    if not reranked:
        k = max(config.min_k, min(config.max_k, fits))
        return ranked[:k], False
    scores = [c.rerank_score or 0.0 for c in ranked]
    above_min = sum(1 for s in scores if s >= config.min_score)
    near_top = sum(1 for s in scores if s >= config.relative_to_top * scores[0])
    k = max(config.min_k, min(above_min, near_top, fits))
    k = min(k, config.max_k)
    return ranked[:k], above_min == 0
