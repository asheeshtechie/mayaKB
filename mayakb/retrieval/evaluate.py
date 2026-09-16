"""Evaluation harness: score search quality and reranker cost on labelled queries.

queries.jsonl, one per line:
  {"query": "...", "relevant_chunk_ids": ["..."], "relevant_doc_ids": ["..."], "filters": {...}}
Chunk ids are used when given; otherwise document ids.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from mayakb.retrieval.search import Searcher, SearchOptions, SearchResults

VARIANTS: dict[str, SearchOptions] = {
    "fixed_beam": SearchOptions(beam="fixed"),
    "adaptive_beam": SearchOptions(beam="adaptive"),
    "adaptive_no_rerank": SearchOptions(beam="adaptive", rerank=False),
    "dense_only": SearchOptions(beam="adaptive", retrievers="dense"),
    "keyword_only": SearchOptions(beam="adaptive", retrievers="keyword"),
}


class EvalQuery(BaseModel):
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None


class EvalReport(BaseModel):
    variant: str
    queries: int
    beam_recall: float
    avg_beam_size: float
    recall_at_k: float
    ndcg_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float


def load_queries(path: str | Path) -> list[EvalQuery]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalQuery.model_validate(json.loads(line)) for line in lines if line.strip()]


def recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    return next((1.0 / rank for rank, hit in enumerate(flags, start=1) if hit), 0.0)


def ndcg_at_10(flags: list[bool], relevant_count: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(flags[:10], start=1) if hit)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_count, 10) + 1))
    return dcg / ideal if ideal else 0.0


def _judge(q: EvalQuery, results: SearchResults) -> tuple[list[bool], set[str], list[str], list[str]]:
    """Per-result relevance flags, the relevant set, retrieved ids and beam ids (same id kind)."""
    if q.relevant_chunk_ids:
        relevant = set(q.relevant_chunk_ids)
        flags = [bool(set(c.chunk_ids) & relevant) for c in results.chunks]
        retrieved = [cid for c in results.chunks for cid in c.chunk_ids]
        beam = results.trace["beam"]["chunk_ids"]
    else:
        relevant = set(q.relevant_doc_ids)
        flags = [c.doc_id in relevant for c in results.chunks]
        retrieved = [c.doc_id for c in results.chunks]
        beam = results.trace["beam"]["doc_ids"]
    return flags, relevant, retrieved, beam


def evaluate(
    searcher: Searcher, queries: list[EvalQuery], options: SearchOptions, variant: str = "custom"
) -> EvalReport:
    if not queries:
        raise ValueError("no queries to evaluate")
    beam_recalls, beam_sizes, recalls, ndcgs, rrs, latencies = [], [], [], [], [], []
    for q in queries:
        start = time.perf_counter()
        results = searcher.search(q.query, filters=q.filters, options=options)
        latencies.append((time.perf_counter() - start) * 1000)
        flags, relevant, retrieved, beam = _judge(q, results)
        beam_recalls.append(recall(beam, relevant))
        beam_sizes.append(results.trace["counts"]["beam"])
        recalls.append(recall(retrieved, relevant))
        ndcgs.append(ndcg_at_10(flags, len(relevant)))
        rrs.append(reciprocal_rank(flags))
    return EvalReport(
        variant=variant,
        queries=len(queries),
        beam_recall=round(float(np.mean(beam_recalls)), 4),
        avg_beam_size=round(float(np.mean(beam_sizes)), 2),
        recall_at_k=round(float(np.mean(recalls)), 4),
        ndcg_at_10=round(float(np.mean(ndcgs)), 4),
        mrr=round(float(np.mean(rrs)), 4),
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2),
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2),
    )


def compare(searcher: Searcher, queries: list[EvalQuery]) -> list[EvalReport]:
    """Run every variant in VARIANTS on the same queries."""
    return [evaluate(searcher, queries, options, name) for name, options in VARIANTS.items()]
