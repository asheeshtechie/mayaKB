import json

import pytest
from typer.testing import CliRunner

from mayakb import cli, wiring
from mayakb.core.config import Settings
from mayakb.storage.artifacts import FileArtifactStore
from mayakb.storage.joblog import InMemoryJobLog
from mayakb.storage.search.memory import InMemorySearchBackend
from tests.fakes import make_kb

runner = CliRunner()

DOCS = {
    "mem://docs/refunds.md": "# Refunds\n\nRefunds are issued within 30 days.",
    "mem://docs/shipping.md": "# Shipping\n\nOrders ship from Berlin within two days.",
    "mem://docs/bad.md": "# Broken\n\nThe picky parser refuses this one.",
}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """An in-memory Runtime that every command gets instead of Postgres."""
    rt = wiring.Runtime(
        settings=Settings(data_dir=tmp_path),
        kb=make_kb(DOCS, parse={"parser": "picky"}, retrieval={"reranker": {"reranker": "overlap"}}),
        store=FileArtifactStore(tmp_path),
        joblog=InMemoryJobLog(),
        backend=InMemorySearchBackend(),
    )
    monkeypatch.setattr(wiring, "build_runtime", lambda kb_name, settings=None, pool=None: rt)
    return rt


def invoke(*args):
    return runner.invoke(cli.app, list(args))


def test_ingest_reports_each_document_and_fails_on_errors(runtime):
    result = invoke("ingest", "--kb", "test-kb")
    assert result.exit_code == 1
    assert "succeeded mem://docs/refunds.md" in result.output
    assert "failed    mem://docs/bad.md  [understand_document]" in result.output
    assert "done: 2 ingested, 0 unchanged, 1 failed" in result.output


def test_search_prints_citations_json_and_trace(runtime):
    invoke("ingest", "--kb", "test-kb")
    text = invoke("search", "refunds issued", "--kb", "test-kb", "--trace")
    assert text.exit_code == 0, text.output
    assert "[1] score=1.00  mem://docs/refunds.md  Refunds" in text.output
    assert '"counts"' in text.output

    as_json = invoke("search", "orders", "--kb", "test-kb", "--entity", "Berlin", "--json")
    data = json.loads(as_json.output)
    assert [c["citation"]["source_uri"] for c in data["chunks"]] == ["mem://docs/shipping.md"]


def test_run_jobs_and_retry(runtime, monkeypatch):
    invoke("ingest", "--kb", "test-kb")
    failed = invoke("jobs", "--kb", "test-kb", "--status", "failed")
    assert "understand_document" in failed.output
    job_id = failed.output.split()[0]

    # fix the cause (switch to the local parser), then retry the failed job
    runtime.kb = make_kb(DOCS, parse={"parser": "local"}, retrieval={"reranker": {"reranker": "overlap"}})
    monkeypatch.setattr(wiring, "find_job", lambda job_id: (runtime.joblog.get(job_id), None))
    retried = invoke("retry", job_id)
    assert retried.exit_code == 0, retried.output
    assert json.loads(retried.output)["status"] == "succeeded"

    ran = invoke("run", "read_source", "mem://docs/refunds.md", "--kb", "test-kb")
    assert json.loads(ran.output)["stage"] == "read_source"


def test_eval_prints_a_row_per_variant(runtime, tmp_path):
    invoke("ingest", "--kb", "test-kb")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query": "refunds", "relevant_doc_ids": ["x"]}) + "\n")
    result = invoke("eval", "--kb", "test-kb", "--queries", str(queries))
    assert result.exit_code == 0, result.output
    assert "fixed_beam" in result.output and "adaptive_beam" in result.output


def test_errors_are_one_line_messages(monkeypatch):
    def missing(kb_name, settings=None, pool=None):
        raise FileNotFoundError("no config for knowledge base 'nope' at configs/nope.yaml")

    monkeypatch.setattr(wiring, "build_runtime", missing)
    result = invoke("search", "x", "--kb", "nope")
    assert result.exit_code == 1
    assert "error: no config for knowledge base 'nope'" in result.output


def test_db_init(monkeypatch):
    monkeypatch.setenv("MAYAKB_DATABASE_URL", "postgresql://u:p@db/mayakb")
    monkeypatch.setattr(cli, "apply_migrations", lambda url: ["001_init"])
    assert "applied migrations: 001_init" in invoke("db", "init").output
    monkeypatch.setattr(cli, "apply_migrations", lambda url: [])
    assert "up to date" in invoke("db", "init").output
    monkeypatch.delenv("MAYAKB_DATABASE_URL")
    result = invoke("db", "init")
    assert result.exit_code == 1 and "MAYAKB_DATABASE_URL is not set" in result.output
