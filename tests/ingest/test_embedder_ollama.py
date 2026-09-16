import json
import os

import httpx
import pytest

from mayakb.core import registry
from mayakb.ingest.embedders.base import EmbeddingError
from mayakb.ingest.embedders.ollama import OllamaEmbedder


def fake_ollama(responses: list[int] | None = None, drop_one: bool = False):
    """Echo server: returns [len(input), 1.0] per input. `responses` forces status codes in order."""
    requests: list[dict] = []
    codes = iter(responses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        requests.append(body)
        code = next(codes, 200)
        if code != 200:
            return httpx.Response(code, json={"error": "nope"})
        vectors = [[float(len(text)), 1.0] for text in body["input"]]
        return httpx.Response(200, json={"model": body["model"], "embeddings": vectors[:-1] if drop_one else vectors})

    return httpx.MockTransport(handler), requests


def make(transport, **kwargs) -> OllamaEmbedder:
    return OllamaEmbedder(base_url="http://ollama:11434", transport=transport, sleep=lambda _: None, **kwargs)


def test_nomic_prefixes_are_applied():
    transport, requests = fake_ollama()
    embedder = make(transport, model="nomic-embed-text:latest")
    embedder.embed_documents(["refunds"])
    embedder.embed_query("refunds?")
    assert requests[0]["input"] == ["search_document: refunds"]
    assert requests[1]["input"] == ["search_query: refunds?"]
    assert requests[0]["model"] == "nomic-embed-text:latest"


def test_other_models_have_no_prefix_by_default():
    transport, requests = fake_ollama()
    make(transport, model="mxbai-embed-large").embed_documents(["x"])
    assert requests[0]["input"] == ["x"]


def test_batches_requests_and_keeps_order():
    transport, requests = fake_ollama()
    vectors = make(transport, model="all-minilm", batch_size=2).embed_documents(["a", "bb", "ccc"])
    assert [len(r["input"]) for r in requests] == [2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_retries_server_errors():
    transport, requests = fake_ollama(responses=[503, 200])
    assert make(transport, model="all-minilm").embed_query("x") == [1.0, 1.0]
    assert len(requests) == 2


def test_missing_model_explains_how_to_pull_it():
    transport, _ = fake_ollama(responses=[404])
    with pytest.raises(EmbeddingError, match="ollama pull all-minilm"):
        make(transport, model="all-minilm").embed_query("x")


def test_wrong_number_of_vectors_is_an_error():
    transport, _ = fake_ollama(drop_one=True)
    with pytest.raises(EmbeddingError):
        make(transport, model="all-minilm").embed_documents(["a", "b"])


def test_base_url_comes_from_environment(monkeypatch):
    monkeypatch.setenv("MAYAKB_OLLAMA_URL", "http://elsewhere:9999")
    assert str(OllamaEmbedder()._client.base_url) == "http://elsewhere:9999"


def test_registered_as_ollama():
    assert registry.get("embedder", "ollama") is OllamaEmbedder


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("MAYAKB_TEST_OLLAMA_URL"), reason="set MAYAKB_TEST_OLLAMA_URL to call Ollama")
def test_real_ollama():
    embedder = OllamaEmbedder(base_url=os.environ["MAYAKB_TEST_OLLAMA_URL"])
    [vector] = embedder.embed_documents(["hello world"])
    assert len(vector) == 768
