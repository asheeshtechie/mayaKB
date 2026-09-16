from mayakb.retrieval.fusion import Candidate
from mayakb.storage.search.base import Hit


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

