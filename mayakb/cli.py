"""Command line: mayakb db init | ingest | run | jobs | retry | search | eval."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from psycopg_pool import ConnectionPool

from mayakb import wiring
from mayakb.core.config import Settings
from mayakb.ingest.jobs import JobFailed, run_stage
from mayakb.ingest.runner import ingest as run_ingest
from mayakb.retrieval.context import ResultChunk
from mayakb.retrieval.evaluate import compare, load_queries
from mayakb.retrieval.search import SearchOptions
from mayakb.storage.db import DatabaseUnavailable, apply_migrations
from mayakb.storage.search.base import Filters

app = typer.Typer(no_args_is_help=True, add_completion=False, help="mayaKB: ingest documents and search them.")
db_app = typer.Typer(no_args_is_help=True, help="Database commands.")
app.add_typer(db_app, name="db")

KbOption = Annotated[str, typer.Option("--kb", help="Knowledge base name; its config is configs/<kb>.yaml.")]

# Errors worth a one-line message instead of a traceback.
EXPECTED_ERRORS = (DatabaseUnavailable, FileNotFoundError, JobFailed, LookupError, ValueError)


def main() -> None:
    app()


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


@contextmanager
def _runtime(kb: str, pool: ConnectionPool | None = None) -> Iterator[wiring.Runtime]:
    try:
        runtime = wiring.build_runtime(kb, pool=pool)
    except EXPECTED_ERRORS as exc:
        if pool is not None:
            pool.close()
        _fail(str(exc))
    try:
        yield runtime
    except EXPECTED_ERRORS as exc:
        _fail(str(exc))
    finally:
        runtime.close()


@db_app.command("init")
def db_init() -> None:
    """Create or upgrade the database schema. Safe to run repeatedly."""
    settings = Settings.from_env()
    if not settings.database_url:
        _fail("MAYAKB_DATABASE_URL is not set")
    try:
        applied = apply_migrations(settings.database_url)
    except DatabaseUnavailable as exc:
        _fail(str(exc))
    typer.echo(f"applied migrations: {', '.join(applied)}" if applied else "database schema is up to date")


@app.command()
def ingest(kb: KbOption) -> None:
    """Fetch, parse, chunk, tag, embed and index every new or changed document in the source."""
    with _runtime(kb) as rt:
        report = run_ingest(rt.context())
    for doc in report.documents:
        line = f"{doc.status:9} {doc.source_uri}"
        if doc.status == "failed":
            line += f"  [{doc.failed_stage}] {doc.error}"
        typer.echo(line)
    counts = report.counts()
    typer.echo(f"done: {counts['succeeded']} ingested, {counts['skipped']} unchanged, {counts['failed']} failed")
    if counts["failed"]:
        raise typer.Exit(code=1)


@app.command("run")
def run_job(
    stage: Annotated[str, typer.Argument(help="read_source, understand_document, chunk_document, "
                                              "extract_entities, embed_chunks or index_chunks")],
    input_ids: Annotated[list[str], typer.Argument(help="The stage's input ids (a uri for read_source).")],
    kb: KbOption,
) -> None:
    """Run one job, e.g. `mayakb run chunk_document <doc_model_id> --kb docs`."""
    with _runtime(kb) as rt:
        result = run_stage(stage, input_ids, ctx=rt.context())
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def jobs(
    kb: KbOption,
    status: Annotated[str | None, typer.Option(help="pending, running, succeeded, failed or skipped")] = None,
    limit: Annotated[int, typer.Option(help="How many to show, newest first.")] = 50,
) -> None:
    """List recent jobs."""
    with _runtime(kb) as rt:
        records = rt.joblog.list(kb, status=status, limit=limit)
    for r in records:
        typer.echo(f"{r.job_id}  {r.status:9}  {r.stage:20}  attempts={r.attempts}  {r.error or ''}".rstrip())
    if not records:
        typer.echo("no jobs")


@app.command()
def retry(job_id: str) -> None:
    """Run a job again with the same inputs, e.g. after fixing whatever made it fail."""
    try:
        record, pool = wiring.find_job(job_id)
    except EXPECTED_ERRORS as exc:
        _fail(str(exc))
    with _runtime(record.kb, pool=pool) as rt:
        result = run_stage(record.stage, record.input_ids, ctx=rt.context())
    typer.echo(result.model_dump_json(indent=2))


def _describe(rank: int, chunk: ResultChunk) -> str:
    c = chunk.citation
    pages = ""
    if c.page_start is not None:
        pages = f" p.{c.page_start}" + (f"-{c.page_end}" if c.page_end not in (None, c.page_start) else "")
    score = f"{chunk.rerank_score:.2f}" if chunk.rerank_score is not None else "-"
    text = " ".join(chunk.text.split())
    snippet = text[:300] + ("…" if len(text) > 300 else "")
    return f"[{rank}] score={score}  {c.source_uri}{pages}  {' > '.join(c.section_path)}\n    {snippet}"


@app.command()
def search(
    query: str,
    kb: KbOption,
    source_prefix: Annotated[str | None, typer.Option(help="Only documents whose uri starts with this.")] = None,
    entity: Annotated[list[str] | None, typer.Option(help="Only chunks tagged with this entity. Repeatable.")] = None,
    principal: Annotated[list[str] | None, typer.Option(help="Reserved: search as this principal. Repeatable.")] = None,
    beam: Annotated[str | None, typer.Option(help="adaptive or fixed (default: from the KB config)")] = None,
    no_rerank: Annotated[bool, typer.Option("--no-rerank", help="Skip the reranker.")] = False,
    trace: Annotated[bool, typer.Option("--trace", help="Also print counts, beam and timings.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the whole result as JSON.")] = False,
) -> None:
    """Hybrid search with the dynamic beam and reranker."""
    with _runtime(kb) as rt:
        options = SearchOptions(beam=beam, rerank=not no_rerank)
        filters = Filters(source_prefix=source_prefix, entities=entity or None)
        results = rt.searcher().search(query, filters=filters, options=options, principals=principal or None)
    if as_json:
        typer.echo(results.model_dump_json(indent=2))
        return
    if results.low_confidence:
        typer.echo("(low confidence: nothing scored above the rerank threshold)")
    for rank, chunk in enumerate(results.chunks, start=1):
        typer.echo(_describe(rank, chunk))
    if not results.chunks:
        typer.echo("no results")
    if trace:
        typer.echo(json.dumps(results.trace, indent=2))


@app.command("eval")
def eval_command(
    kb: KbOption,
    queries: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Labelled queries, one JSON per line.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the reports as JSON.")] = False,
) -> None:
    """Compare fixed vs adaptive beam, reranker on/off, and dense/keyword/hybrid retrieval."""
    with _runtime(kb) as rt:
        reports = compare(rt.searcher(), load_queries(queries))
    if as_json:
        typer.echo(json.dumps([r.model_dump() for r in reports], indent=2))
        return
    typer.echo(f"{'variant':20} {'beam_recall':>11} {'avg_beam':>8} {'recall@k':>8} {'ndcg@10':>7} {'mrr':>6} "
               f"{'p50_ms':>7} {'p95_ms':>7}")
    for r in reports:
        typer.echo(f"{r.variant:20} {r.beam_recall:>11.3f} {r.avg_beam_size:>8.1f} {r.recall_at_k:>8.3f} "
                   f"{r.ndcg_at_10:>7.3f} {r.mrr:>6.3f} {r.latency_p50_ms:>7.1f} {r.latency_p95_ms:>7.1f}")
