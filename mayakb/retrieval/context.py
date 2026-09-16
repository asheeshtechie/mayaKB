"""Turn the final candidates into result chunks: drop duplicates, merge neighbours, attach citations."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mayakb.retrieval.fusion import Candidate


class Citation(BaseModel):
    source_uri: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)


class ResultChunk(BaseModel):
    chunk_ids: list[str]  # more than one when neighbouring chunks were merged
    doc_id: str
    text: str
    rerank_score: float | None = None
    fused_score: float
    citation: Citation


@dataclass
class _Group:
    members: list[Candidate] = field(default_factory=list)

    @property
    def first(self) -> Candidate:
        return self.members[0]

    def positions(self) -> list[int]:
        return [m.hit.position for m in self.members]

    def adjacent(self, c: Candidate) -> bool:
        return (
            c.hit.doc_id == self.first.hit.doc_id
            and c.hit.section_path == self.first.hit.section_path
            and (c.hit.position == min(self.positions()) - 1 or c.hit.position == max(self.positions()) + 1)
        )


def _dedupe_key(text: str) -> str:
    return " ".join(text.lower().split())


def assemble(selected: list[Candidate]) -> list[ResultChunk]:
    """`selected` is best first. Results keep the rank of their best member."""
    seen: set[str] = set()
    groups: list[_Group] = []
    for c in selected:
        key = _dedupe_key(c.hit.text)
        if key in seen:
            continue
        seen.add(key)
        group = next((g for g in groups if g.adjacent(c)), None)
        if group is None:
            groups.append(_Group([c]))
        else:
            group.members.append(c)

    results = []
    for group in groups:
        members = sorted(group.members, key=lambda m: m.hit.position)
        pages_start = [m.hit.page_start for m in members if m.hit.page_start is not None]
        pages_end = [m.hit.page_end for m in members if m.hit.page_end is not None]
        rerank = [m.rerank_score for m in members if m.rerank_score is not None]
        top = group.first.hit
        results.append(ResultChunk(
            chunk_ids=[m.chunk_id for m in members],
            doc_id=top.doc_id,
            text="\n\n".join(m.hit.text for m in members),
            rerank_score=max(rerank) if rerank else None,
            fused_score=max(m.fused_score for m in members),
            citation=Citation(
                source_uri=top.source_uri,
                title=top.title,
                page_start=min(pages_start) if pages_start else None,
                page_end=max(pages_end) if pages_end else None,
                section_path=top.section_path,
            ),
        ))
    return results
