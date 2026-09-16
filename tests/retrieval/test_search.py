import pytest

from mayakb.core.config import BeamConfig, CutConfig
from mayakb.retrieval.search import IndexNotFound, Searcher, SearchOptions
from mayakb.storage.search.memory import InMemorySearchBackend
from tests.fakes import HashEmbedder, OverlapReranker, make_kb
from tests.retrieval.helpers import DOCS, make_searcher


def test_finds_the_best_chunk_with_a_citation(corpus):
    results = make_searcher(corpus).search("refunds issued")
    top = results.chunks[0]
    assert top.citation.source_uri == "mem://docs/refunds.md"
    assert top.citation.section_path == ["Refunds"]
    assert top.rerank_score == 1.0
    assert results.trace["dense"] and results.trace["keyword"] and results.trace["reranked"]
    assert results.low_confidence is False


def test_only_the_beam_reaches_the_reranker(corpus):
    results = make_searcher(corpus).search("refunds issued")
    [(_, reranked_texts)] = OverlapReranker.calls
    counts = results.trace["counts"]
    assert len(reranked_texts) == counts["beam"] == counts["reranked"]
    assert counts["beam"] <= 3 < counts["fused"]
    assert sum(results.trace["beam"]["kept_by"].values()) == counts["beam"]


def test_filters_are_applied(corpus):
    searcher = make_searcher(corpus)
    archived = searcher.search("refunds", filters={"source_prefix": "mem://archive/"})
    assert {c.citation.source_uri for c in archived.chunks} == {"mem://archive/old-refunds.md"}
    berlin = searcher.search("orders", filters={"entities": ["BERLIN"]})
    assert [c.citation.section_path for c in berlin.chunks] == [["Shipping"]]


def test_embedder_failure_falls_back_to_keyword_search(corpus):
    class BrokenEmbedder(HashEmbedder):
        def embed_query(self, text):
            raise ConnectionError("ollama is down")

    results = make_searcher(corpus, embedder=BrokenEmbedder()).search("refunds issued")
    assert results.trace["dense"] is False
    assert "ollama is down" in results.trace["dense_error"]
    assert results.trace["counts"]["dense"] == 0
    assert results.chunks[0].citation.section_path == ["Refunds"]


def test_reranker_failure_keeps_fused_order(corpus):
    results = make_searcher(corpus, reranker=OverlapReranker(fail=True)).search("refunds issued")
    assert results.trace["reranked"] is False
    assert "reranker is down" in results.trace["rerank_error"]
    assert results.chunks
    assert all(c.rerank_score is None for c in results.chunks)


def test_rerank_can_be_turned_off(corpus):
    results = make_searcher(corpus).search("refunds", options=SearchOptions(rerank=False))
    assert OverlapReranker.calls == []
    assert results.trace["reranked"] is False


def test_fallback_widens_only_when_enabled_and_nothing_scores_well(corpus):
    strict_cut = CutConfig(min_k=1, max_k=5, min_score=1.0, relative_to_top=0.5)
    narrow = BeamConfig(min_beam=1, max_beam=1)
    make_searcher(corpus, beam=narrow, cut=strict_cut).search("refunds zebra")
    assert len(OverlapReranker.calls) == 1

    OverlapReranker.calls.clear()
    results = make_searcher(
        corpus, beam=narrow.model_copy(update={"fallback_widen": True}), cut=strict_cut
    ).search("refunds zebra")
    assert len(OverlapReranker.calls) == 2
    assert results.trace["fallback_used"] is True
    assert results.trace["counts"]["reranked"] == 2
    assert results.low_confidence is True


def test_retriever_and_beam_options(corpus):
    searcher = make_searcher(corpus)
    keyword_only = searcher.search("refunds", options=SearchOptions(retrievers="keyword"))
    assert (keyword_only.trace["dense"], keyword_only.trace["counts"]["dense"]) == (False, 0)
    dense_only = searcher.search("refunds", options=SearchOptions(retrievers="dense", beam="fixed"))
    assert (dense_only.trace["keyword"], dense_only.trace["counts"]["keyword"]) == (False, 0)
    assert dense_only.trace["beam"]["policy"] == "fixed"


def test_no_matches_is_an_empty_result_not_an_error(corpus):
    results = make_searcher(corpus).search("zzzz qqqq", options=SearchOptions(retrievers="keyword"))
    assert results.chunks == []
    assert results.low_confidence is False


def test_bad_queries_and_missing_index(corpus):
    with pytest.raises(ValueError):
        make_searcher(corpus).search("   ")
    kb, _ = corpus
    with pytest.raises(IndexNotFound, match="mayakb ingest"):
        Searcher(kb, InMemorySearchBackend(), HashEmbedder(), None).search("refunds")


def test_from_config_builds_implementations_and_survives_a_broken_reranker(corpus):
    kb, backend = corpus
    searcher = Searcher.from_config(kb, backend)
    assert isinstance(searcher.reranker, OverlapReranker)
    assert isinstance(searcher.embedder, HashEmbedder)

    broken = make_kb(DOCS, retrieval={"reranker": {"reranker": "overlap", "no_such_option": 1}})
    searcher = Searcher.from_config(broken, backend)
    assert searcher.reranker is None
    results = searcher.search("refunds")
    assert results.chunks
    assert "no_such_option" in results.trace["rerank_error"]
