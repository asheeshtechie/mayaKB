from mayakb.core.config import BeamConfig, CutConfig
from mayakb.retrieval.beam import AdaptiveBeam, FixedBeam, cut, make_beam
from tests.retrieval.helpers import candidate


def fused_list():
    """Fused order a..h. Best dense 0.80, best keyword 10.0."""
    return [
        candidate("a", dense_rank=1, dense_score=0.80, keyword_rank=1, keyword_score=10.0),  # agreement
        candidate("b", dense_rank=2, dense_score=0.75),                                       # dense (>= 0.72)
        candidate("c", keyword_rank=2, keyword_score=7.0),                                    # keyword (>= 6.0)
        candidate("d", dense_rank=3, dense_score=0.60),                                       # dropped
        candidate("e", keyword_rank=3, keyword_score=2.0),                                    # dropped
        candidate("f", dense_rank=30, dense_score=0.50, keyword_rank=25, keyword_score=1.0),  # outside agree_top
        candidate("g", dense_rank=4, dense_score=0.10),
        candidate("h", keyword_rank=4, keyword_score=0.5),
    ]


def ids(candidates):
    return [c.chunk_id for c in candidates]


def test_adaptive_keeps_only_confident_candidates():
    selection = AdaptiveBeam(min_beam=1, max_beam=30, agree_top=20, dense_rel=0.9, keyword_rel=0.6).select(fused_list())
    assert ids(selection.beam) == ["a", "b", "c"]
    assert selection.kept_by == {"agreement": 1, "dense": 1, "keyword": 1}
    assert ids(selection.rest) == ["d", "e", "f", "g", "h"]


def test_adaptive_tops_up_to_min_beam_in_fused_order():
    selection = AdaptiveBeam(min_beam=5).select(fused_list())
    assert ids(selection.beam) == ["a", "b", "c", "d", "e"]
    assert selection.kept_by["top_up"] == 2


def test_adaptive_respects_max_beam():
    many = [candidate(f"c{i:02d}", dense_rank=i + 1, dense_score=0.9) for i in range(50)]
    selection = AdaptiveBeam(min_beam=5, max_beam=30).select(many)
    assert len(selection.beam) == 30
    assert ids(selection.beam) == ids(many[:30])


def test_adaptive_handles_one_sided_and_empty_input():
    keyword_only = [
        candidate("k1", keyword_rank=1, keyword_score=3.0),
        candidate("k2", keyword_rank=2, keyword_score=0.5),
    ]
    assert ids(AdaptiveBeam(min_beam=1).select(keyword_only).beam) == ["k1"]
    assert AdaptiveBeam().select([]).beam == []


def test_fixed_beam_and_factory():
    assert ids(FixedBeam(size=2).select(fused_list()).beam) == ["a", "b"]
    assert isinstance(make_beam(BeamConfig(policy="fixed", size=3)), FixedBeam)
    assert isinstance(make_beam(BeamConfig()), AdaptiveBeam)


def scored(*scores, words=10):
    return [candidate(f"r{i}", rerank_score=s, text=" ".join(["w"] * words)) for i, s in enumerate(scores)]


CUT = CutConfig(min_k=2, max_k=10, min_score=0.3, relative_to_top=0.5)


def test_cut_by_min_score_and_relative_to_top():
    selected, low = cut(scored(0.9, 0.8, 0.5, 0.35, 0.1), CUT, 10_000, reranked=True)
    assert ids(selected) == ["r0", "r1", "r2"]  # 0.35 is below 0.5 * 0.9
    assert low is False


def test_cut_by_token_budget_but_never_below_min_k():
    selected, _ = cut(scored(0.9, 0.9, 0.9, 0.9, words=100), CUT, 250, reranked=True)
    assert len(selected) == 2
    selected, _ = cut(scored(0.9, 0.9, 0.9, words=100), CUT, 50, reranked=True)
    assert len(selected) == 2  # min_k wins even over budget


def test_cut_low_confidence_returns_min_k():
    selected, low = cut(scored(0.2, 0.1, 0.05), CUT, 10_000, reranked=True)
    assert ids(selected) == ["r0", "r1"]
    assert low is True


def test_cut_max_k():
    selected, _ = cut(scored(*([0.9] * 20)), CUT, 10_000, reranked=True)
    assert len(selected) == 10


def test_cut_without_reranker_uses_budget_and_max_k():
    selected, low = cut(scored(*([None] * 20), words=100), CUT, 450, reranked=False)
    assert len(selected) == 4
    assert low is False
    assert cut([], CUT, 100, reranked=True) == ([], False)
