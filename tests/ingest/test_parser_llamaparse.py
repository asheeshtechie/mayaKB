import json
import os

import httpx
import pytest

from mayakb.core import registry
from mayakb.core.contracts import RawDocument
from mayakb.ingest.parsers.llamaparse import LlamaParseError, LlamaParseParser, extract_pages

RAW = RawDocument(
    id="d1",
    source_uri="s3://docs/q2.pdf",
    connector="s3",
    mime="application/pdf",
    metadata={"filename": "q2.pdf"},
)


def fake_api(statuses: list[str], pages: dict | None = None):
    """A MockTransport that walks through `statuses` on each status poll."""
    seen: list[httpx.Request] = []
    polls = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.method == "POST" and request.url.path == "/api/v2/parse/upload":
            return httpx.Response(200, json={"id": "job-1", "status": "PENDING"})
        if request.url.path == "/api/v2/parse/job-1":
            if request.url.params.get("expand") == "markdown":
                return httpx.Response(200, json=pages)
            status = next(polls)
            return httpx.Response(200, json={"job": {"id": "job-1", "status": status, "error_message": "bad file"}})
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


PAGES = {
    "job": {"id": "job-1", "status": "COMPLETED"},
    "markdown": {
        "pages": [
            {"page_number": 1, "markdown": "# Q2 Report\n\nRevenue grew."},
            {"page_number": 2, "markdown": "## Churn\n\nChurn fell in EMEA."},
        ]
    },
}


def make_parser(transport) -> LlamaParseParser:
    return LlamaParseParser(api_key="test-key", transport=transport, sleep=lambda _: None)


def test_parse_uploads_polls_and_builds_pages():
    transport, seen = fake_api(["PENDING", "RUNNING", "COMPLETED"], PAGES)
    doc = make_parser(transport).parse(RAW, b"%PDF")
    assert doc.parser == "llamaparse"
    assert doc.title == "Q2 Report"
    assert doc.metadata["llamaparse_job_id"] == "job-1"
    churn = next(e for e in doc.elements if e.text == "Churn fell in EMEA.")
    assert churn.page == 2
    assert churn.section_path == ["Q2 Report", "Churn"]
    upload = seen[0]
    assert b'name="configuration"' in upload.content
    assert json.dumps({"tier": "agentic", "version": "latest"}).encode() in upload.content


def test_failed_job_raises():
    transport, _ = fake_api(["FAILED"], PAGES)
    with pytest.raises(LlamaParseError, match="FAILED"):
        make_parser(transport).parse(RAW, b"%PDF")


def test_times_out():
    transport, _ = fake_api(["PENDING"] * 10, PAGES)
    ticks = iter(range(0, 1000, 400))
    parser = LlamaParseParser(
        api_key="test-key", transport=transport, sleep=lambda _: None, timeout=500, clock=lambda: next(ticks)
    )
    with pytest.raises(TimeoutError):
        parser.parse(RAW, b"%PDF")


def test_extract_pages_accepts_both_shapes():
    assert extract_pages(PAGES) == [(1, "# Q2 Report\n\nRevenue grew."), (2, "## Churn\n\nChurn fell in EMEA.")]
    alt = {"job": {"markdown": [{"page": 0, "content": "a"}, {"page": 1, "content": "b"}]}}
    assert extract_pages(alt) == [(1, "a"), (2, "b")]
    with pytest.raises(LlamaParseError):
        extract_pages({"job": {"status": "COMPLETED"}})


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLAMA_CLOUD_API_KEY"):
        LlamaParseParser()


def test_registered_as_llamaparse():
    assert registry.get("parser", "llamaparse") is LlamaParseParser


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LLAMA_CLOUD_API_KEY"), reason="set LLAMA_CLOUD_API_KEY to call LlamaParse")
def test_real_llamaparse_markdown_document():
    raw = RawDocument(id="d1", source_uri="file:///t.md", connector="local", mime="text/markdown",
                      metadata={"filename": "t.md"})
    doc = LlamaParseParser().parse(raw, b"# Hello\n\nThis is a LlamaParse integration test.")
    assert any("integration test" in e.text for e in doc.elements)
