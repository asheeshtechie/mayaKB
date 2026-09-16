"""LlamaParse (LlamaCloud) parser. Online: the document is uploaded to LlamaIndex's servers.

API: POST /api/v2/parse/upload, then poll GET /api/v2/parse/{job_id}, then fetch per-page
markdown with ?expand=markdown. Check https://developers.llamaindex.ai/llamaparse/ if it changes.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from mayakb.core.contracts import DocModel, RawDocument
from mayakb.core.retry import with_retries
from mayakb.ingest.parsers.base import ElementBuilder, Parser, first_heading
from mayakb.ingest.parsers.markdown import parse_markdown_into

DEFAULT_BASE_URL = "https://api.cloud.llamaindex.ai"


class LlamaParseError(RuntimeError):
    pass


def extract_pages(payload: dict[str, Any]) -> list[tuple[int, str]]:
    """(1-based page number, markdown) from an ?expand=markdown response.

    Documented shape: {"job": {...}, "markdown": {"pages": [{"page_number": 1, "markdown": "..."}]}}.
    Also accepts {"job": {"markdown": [{"page": 0, "content": "..."}]}} (0-based pages).
    """
    markdown = payload.get("markdown")
    if isinstance(markdown, dict) and isinstance(markdown.get("pages"), list):
        return [(int(p["page_number"]), p.get("markdown") or "") for p in markdown["pages"]]
    job_markdown = (payload.get("job") or {}).get("markdown")
    if isinstance(job_markdown, list):
        return [(int(p["page"]) + 1, p.get("content") or "") for p in job_markdown]
    raise LlamaParseError("LlamaParse response has no per-page markdown")


class LlamaParseParser(Parser):
    name = "llamaparse"
    version = "1"

    def __init__(
        self,
        tier: str = "agentic",
        api_version: str = "latest",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not key:
            raise ValueError("LLAMA_CLOUD_API_KEY is not set")
        self.tier = tier
        self.api_version = api_version
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=60.0,
            transport=transport,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        return with_retries(call, sleep=self._sleep)

    def parse(self, raw: RawDocument, data: bytes) -> DocModel:
        job_id = self._upload(raw, data)
        self._wait_until_complete(job_id)
        result = self._request("GET", f"/api/v2/parse/{job_id}", params={"expand": "markdown"})
        builder = ElementBuilder()
        for page_number, markdown in extract_pages(result):
            parse_markdown_into(builder, markdown, page=page_number)
        return DocModel(
            doc_id=raw.id,
            parser=self.name,
            source_uri=raw.source_uri,
            title=first_heading(builder.elements) or raw.metadata.get("filename"),
            acl=raw.acl,
            metadata={"mime": raw.mime, "llamaparse_job_id": job_id},
            elements=builder.elements,
        )

    def _upload(self, raw: RawDocument, data: bytes) -> str:
        body = self._request(
            "POST",
            "/api/v2/parse/upload",
            files={"file": (raw.metadata.get("filename") or "document", data, raw.mime)},
            data={"configuration": json.dumps({"tier": self.tier, "version": self.api_version})},
        )
        job_id = body.get("id") or (body.get("job") or {}).get("id")
        if not job_id:
            raise LlamaParseError(f"upload response has no job id: {body}")
        return job_id

    def _wait_until_complete(self, job_id: str) -> None:
        deadline = self._clock() + self.timeout
        while True:
            body = self._request("GET", f"/api/v2/parse/{job_id}")
            job = body.get("job") or body
            status = job.get("status")
            if status == "COMPLETED":
                return
            if status in {"FAILED", "CANCELLED"}:
                raise LlamaParseError(f"LlamaParse job {job_id} {status}: {job.get('error_message')}")
            if self._clock() > deadline:
                raise TimeoutError(f"LlamaParse job {job_id} still {status} after {self.timeout}s")
            self._sleep(self.poll_interval)
