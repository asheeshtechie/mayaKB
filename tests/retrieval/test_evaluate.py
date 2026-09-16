import json
import math

import pytest

from mayakb.core.ids import content_id
from mayakb.retrieval.evaluate import (
    VARIANTS,
    EvalQuery,
    compare,
    evaluate,
    load_queries,
    ndcg_at_10,
    recall,
    reciprocal_rank,
)
from mayakb.retrieval.search import SearchOptions
from tests.retrieval.helpers import DOCS, make_searcher


def test_metric_math():
    assert recall(["a", "b"], {"a", "c"}) == 0.5
    assert recall([], set()) == 0.0
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([False]) == 0.0
    assert ndcg_at_10([True, False], 1) == 1.0
    assert ndcg_at_10([False, True], 1) == pytest.approx(1 / math.log2(3))


def test_load_queries(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps({"query": "refunds", "relevant_doc_ids": ["d1"]}) + "\n\n")
    assert load_queries(path) == [EvalQuery(query="refunds", relevant_doc_ids=["d1"])]


def test_evaluate_on_a_small_corpus(corpus):
    searcher = make_searcher(corpus)
    refunds_doc = content_id(DOCS["mem://docs/refunds.md"].encode())
    queries = [EvalQuery(query="refunds issued", relevant_doc_ids=[refunds_doc])]
    report = evaluate(searcher, queries, SearchOptions(), "adaptive")
    assert report.queries == 1
    assert report.mrr == 1.0
    assert report.recall_at_k == 1.0
    assert report.beam_recall == 1.0
    assert 1 <= report.avg_beam_size <= 3
    assert report.latency_p95_ms >= report.latency_p50_ms >= 0


def test_compare_runs_every_variant(corpus):
    searcher = make_searcher(corpus)
    reports = compare(searcher, [EvalQuery(query="refunds", relevant_doc_ids=["nope"])])
    assert [r.variant for r in reports] == list(VARIANTS)
    assert all(r.mrr == 0.0 for r in reports)


def test_no_queries_is_an_error(corpus):
    with pytest.raises(ValueError):
        evaluate(make_searcher(corpus), [], SearchOptions())
