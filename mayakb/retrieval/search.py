"""The retrieval pipeline: prepare -> hybrid retrieve -> fuse -> beam -> rerank -> cut -> assemble."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import BaseModel, Field

from mayakb.core import registry
from mayakb.core.config import KBConfig
from mayakb.ingest.embedders.base import Embedder
from mayakb.retrieval.beam import cut, make_beam
from mayakb.retrieval.context import ResultChunk, assemble
from mayakb.retrieval.fusion import Candidate, fuse
from mayakb.retrieval.rerankers.base import Reranker
from mayakb.storage.search.base import Filters, Hit, IndexInfo, SearchBackend


class IndexNotFound(LookupError):
    pass


class SearchOptions(BaseModel):
    beam: Literal["adaptive", "fixed"] | None = None  # None = the KB config's policy
    rerank: bool = True
    token_budget: int | None = None
    retrievers: Literal["hybrid", "dense", "keyword"] = "hybrid"


class SearchResults(BaseModel):
    query: str
    chunks: list[ResultChunk]
    low_confidence: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)


@contextmanager
def _timed(timings: dict[str, float], step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[step] = round((time.perf_counter() - start) * 1000, 2)


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class Searcher:
    def __init__(self, kb: KBConfig, backend: SearchBackend, embedder: Embedder, reranker: Reranker | None,
                 reranker_error: str | None = None):
        self.kb = kb
        self.backend = backend
        self.embedder = embedder
        self.reranker = reranker
        self.reranker_error = reranker_error

    @classmethod
    def from_config(cls, kb: KBConfig, backend: SearchBackend) -> Searcher:
        embedder = registry.create("embedder", kb.embed.name, **kb.embed.params)
        reranker, reranker_error = None, None
        if kb.retrieval.reranker is not None:
            try:
                reranker = registry.create("reranker", kb.retrieval.reranker.name, **kb.retrieval.reranker.params)
            except Exception as exc:  # e.g. model download failed: search still works, unreranked
                reranker_error = _error(exc)
        return cls(kb, backend, embedder, reranker, reranker_error)

    def _index(self) -> IndexInfo:
        index = self.backend.find_index(self.kb.kb, self.kb.embed.name, self.embedder.model)
        if index is None:
            available = [f"{i.embedder}:{i.model}" for i in self.backend.list_indexes(self.kb.kb)]
            raise IndexNotFound(
                f"no index for kb {self.kb.kb!r} with embedder {self.kb.embed.name}:{self.embedder.model}; "
                f"indexed: {available or 'nothing yet (run mayakb ingest)'}"
            )
        return index

    def _retrieve(self, query: str, vector: list[float] | None, use_keyword: bool, depth: int,
                  filters: Filters, index: IndexInfo) -> tuple[list[Hit], list[Hit]]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense = keyword = None
            if vector is not None:
                dense = pool.submit(self.backend.dense_search, vector, depth, filters, index)
            if use_keyword:
                keyword = pool.submit(self.backend.keyword_search, query, depth, filters, index)
            return (dense.result() if dense else []), (keyword.result() if keyword else [])

    def _rerank(self, query: str, candidates: list[Candidate]) -> None:
        scores = self.reranker.score(query, [c.hit.text for c in candidates])
        if len(scores) != len(candidates):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(candidates)} texts")
        for c, s in zip(candidates, scores, strict=True):
            c.rerank_score = s

    def search(
        self,
        query: str,
        filters: Filters | dict[str, Any] | None = None,
        options: SearchOptions | None = None,
        principals: list[str] | None = None,
    ) -> SearchResults:
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        options = options or SearchOptions()
        config = self.kb.retrieval
        beam_config = config.beam if options.beam is None else config.beam.model_copy(update={"policy": options.beam})
        token_budget = options.token_budget or config.token_budget
        filters = filters if isinstance(filters, Filters) else Filters(**(filters or {}))
        if principals is not None:
            filters = filters.model_copy(update={"principals": principals})
        filters = filters.normalized()
        timings: dict[str, float] = {}
        trace: dict[str, Any] = {"dense": False, "keyword": False, "reranked": False, "fallback_used": False,
                                 "timings_ms": timings}

        index = self._index()

        vector = None
        if options.retrievers in ("hybrid", "dense"):
            with _timed(timings, "embed"):
                try:
                    vector = self.embedder.embed_query(query)
                except Exception as exc:  # embedder down: keyword search still works
                    trace["dense_error"] = _error(exc)
        use_keyword = options.retrievers in ("hybrid", "keyword") or vector is None
        with _timed(timings, "retrieve"):
            dense_hits, keyword_hits = self._retrieve(query, vector, use_keyword, beam_config.depth, filters, index)
        trace["dense"], trace["keyword"] = vector is not None, use_keyword

        with _timed(timings, "fuse"):
            candidates = fuse(dense_hits, keyword_hits, config)
        with _timed(timings, "beam"):
            selection = make_beam(beam_config).select(candidates)

        ranked = list(selection.beam)
        reranked_count = 0
        if options.rerank and self.reranker is not None and ranked:
            with _timed(timings, "rerank"):
                try:
                    self._rerank(query, ranked)
                    reranked_count = len(ranked)
                    nothing_good = all(c.rerank_score < config.cut.min_score for c in ranked)
                    if beam_config.fallback_widen and nothing_good and selection.rest:
                        extra = selection.rest[: beam_config.max_beam]
                        self._rerank(query, extra)
                        reranked_count += len(extra)
                        ranked += extra
                        trace["fallback_used"] = True
                    ranked.sort(key=lambda c: -c.rerank_score)
                    trace["reranked"] = True
                except Exception as exc:  # reranker failed: keep fused order
                    trace["rerank_error"] = _error(exc)
                    trace["fallback_used"] = False
                    for c in candidates:
                        c.rerank_score = None
                    ranked, reranked_count = list(selection.beam), 0
        elif options.rerank and self.reranker_error:
            trace["rerank_error"] = self.reranker_error

        with _timed(timings, "cut"):
            selected, low_confidence = cut(ranked, config.cut, token_budget, reranked=trace["reranked"])
        chunks = assemble(selected)

        trace["counts"] = {
            "dense": len(dense_hits), "keyword": len(keyword_hits), "fused": len(candidates),
            "beam": len(selection.beam), "reranked": reranked_count, "final": len(chunks),
        }
        trace["beam"] = {
            "policy": beam_config.policy,
            "kept_by": selection.kept_by,
            "chunk_ids": [c.chunk_id for c in selection.beam],
            "doc_ids": sorted({c.hit.doc_id for c in selection.beam}),
        }
        return SearchResults(query=query, chunks=chunks, low_confidence=low_confidence, trace=trace)
