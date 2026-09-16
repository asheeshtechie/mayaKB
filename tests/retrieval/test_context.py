from mayakb.retrieval.context import assemble
from tests.retrieval.helpers import candidate


def test_merges_adjacent_chunks_of_the_same_section():
    selected = [
        candidate("c5", rerank_score=0.9, position=5, text="middle", page=3),
        candidate("c4", rerank_score=0.6, position=4, text="before", page=2),
        candidate("x1", rerank_score=0.5, position=0, doc_id="d2", text="other doc"),
        candidate("c6", rerank_score=0.4, position=6, text="after", page=3),
    ]
    results = assemble(selected)
    assert [r.chunk_ids for r in results] == [["c4", "c5", "c6"], ["x1"]]
    merged = results[0]
    assert merged.text == "before\n\nmiddle\n\nafter"
    assert merged.rerank_score == 0.9
    assert (merged.citation.page_start, merged.citation.page_end) == (2, 3)
    assert merged.citation.source_uri == "s3://docs/d1.md"
    assert merged.citation.title == "D1"


def test_does_not_merge_across_sections_or_gaps():
    selected = [
        candidate("a", position=1, section_path=["A"]),
        candidate("b", position=2, section_path=["B"]),
        candidate("c", position=4, section_path=["A"]),
    ]
    assert [r.chunk_ids for r in assemble(selected)] == [["a"], ["b"], ["c"]]


def test_drops_near_duplicate_text():
    selected = [
        candidate("a", position=1, doc_id="d1", text="Refunds take 30 days."),
        candidate("b", position=9, doc_id="d2", text="refunds  take 30 days."),
    ]
    assert [r.chunk_ids for r in assemble(selected)] == [["a"]]


def test_empty():
    assert assemble([]) == []
