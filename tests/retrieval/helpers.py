from mayakb.retrieval.fusion import Candidate
from mayakb.retrieval.search import Searcher
from mayakb.storage.search.base import Hit
from tests.fakes import HashEmbedder, OverlapReranker

DOCS = {
    "mem://docs/refunds.md": (
        "# Refunds\n\nRefunds are issued within 30 days of the return.\n\n"
        "# Warranty\n\nThe warranty covers manufacturing defects for two years."
    ),
    "mem://docs/shipping.md": (
        "# Shipping\n\nOrders ship from Berlin within two days.\n\n"
        "# Returns\n\nReturns need the original receipt."
    ),
    "mem://archive/old-refunds.md": "# Old refund rules\n\nRefunds used to take 60 days.",
}


def hit(chunk_id: str, score: float = 1.0, text: str | None = None, doc_id: str = "d1", position: int = 0,
        section_path: list[str] | None = None, page: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id=doc_id, position=position, text=text or f"text of {chunk_id}", score=score,
        source_uri=f"s3://docs/{doc_id}.md", title=doc_id.upper(), page_start=page, page_end=page,
        section_path=section_path if section_path is not None else ["S"],
    )


def candidate(chunk_id: str, *, dense_rank=None, dense_score=None, keyword_rank=None, keyword_score=None,
              rerank_score=None, text=None, **hit_kwargs) -> Candidate:
    return Candidate(
        hit=hit(chunk_id, text=text, **hit_kwargs), fused_score=0.0, dense_rank=dense_rank, dense_score=dense_score,
        keyword_rank=keyword_rank, keyword_score=keyword_score, rerank_score=rerank_score,
    )


def make_searcher(corpus, embedder=None, reranker=None, **retrieval) -> Searcher:
    """A Searcher over the `corpus` fixture; keyword args override retrieval config fields."""
    kb, backend = corpus
    if retrieval:
        kb = kb.model_copy(update={"retrieval": kb.retrieval.model_copy(update=retrieval)})
    return Searcher(kb, backend, embedder or HashEmbedder(), reranker or OverlapReranker())
