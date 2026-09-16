import pytest

from mayakb.core.config import RetrievalConfig
from mayakb.retrieval.fusion import fuse, rrf, weighted
from tests.retrieval.helpers import hit


def test_rrf_rewards_chunks_found_by_both_retrievers():
    dense = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    keyword = [hit("c", 5.0), hit("d", 4.0)]
    fused = rrf(dense, keyword, k=60)
    assert [c.chunk_id for c in fused] == ["c", "a", "b", "d"]  # b and d tie; b wins on chunk_id
    c = fused[0]
    assert c.fused_score == pytest.approx(1 / 63 + 1 / 61)
    assert (c.dense_rank, c.dense_score, c.keyword_rank, c.keyword_score) == (3, 0.7, 1, 5.0)
    assert fused[1].keyword_rank is None


def test_rrf_ties_break_on_best_rank_then_chunk_id():
    fused = rrf([hit("z"), hit("y")], [hit("x"), hit("w")], k=60)
    assert [c.chunk_id for c in fused] == ["x", "z", "w", "y"]


def test_weighted_uses_normalized_scores():
    dense = [hit("a", 0.9), hit("b", 0.5)]
    keyword = [hit("b", 10.0), hit("c", 2.0)]
    fused = weighted(dense, keyword, dense_weight=0.5)
    scores = {c.chunk_id: c.fused_score for c in fused}
    assert scores == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.0})


def test_empty_inputs():
    assert rrf([], []) == []
    assert weighted([], [hit("a", 3.0)])[0].fused_score == pytest.approx(0.4)


def test_fuse_follows_config():
    dense, keyword = [hit("a", 0.9), hit("b", 0.1)], [hit("b", 1.0)]
    assert fuse(dense, keyword, RetrievalConfig(fusion="rrf"))[0].chunk_id == "b"
    assert fuse(dense, keyword, RetrievalConfig(fusion="weighted", dense_weight=1.0))[0].chunk_id == "a"
