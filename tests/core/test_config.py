from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mayakb.core.config import KBConfig, Settings

ROOT = Path(__file__).resolve().parents[2]

MINIMAL = {
    "kb": "finance-docs",
    "source": {"connector": "s3", "bucket": "acme-docs", "prefix": "reports/"},
    "parse": {"parser": "local"},
    "chunk": {"strategy": "by_section", "max_tokens": 512, "overlap_tokens": 50},
    "embed": {"embedder": "ollama", "model": "nomic-embed-text"},
}


def test_sections_become_name_and_params():
    kb = KBConfig.model_validate(MINIMAL)
    assert kb.source.name == "s3"
    assert kb.source.params == {"bucket": "acme-docs", "prefix": "reports/"}
    assert kb.chunk.name == "by_section"
    assert kb.chunk.params == {"max_tokens": 512, "overlap_tokens": 50}


def test_defaults():
    kb = KBConfig.model_validate(MINIMAL)
    assert kb.ner is None
    assert kb.index.name == "postgres"
    assert kb.retrieval.beam.policy == "adaptive"
    assert kb.retrieval.beam.max_beam == 30
    assert kb.retrieval.cut.min_score == 0.3
    assert kb.retrieval.reranker.name == "cross_encoder"


def test_retrieval_overrides_and_reranker_section():
    data = dict(MINIMAL, retrieval={
        "beam": {"policy": "fixed", "size": 12},
        "reranker": {"reranker": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3"},
    })
    kb = KBConfig.model_validate(data)
    assert kb.retrieval.beam.policy == "fixed"
    assert kb.retrieval.beam.size == 12
    assert kb.retrieval.reranker.params == {"model": "BAAI/bge-reranker-v2-m3"}


def test_reranker_can_be_disabled():
    kb = KBConfig.model_validate(dict(MINIMAL, retrieval={"reranker": None}))
    assert kb.retrieval.reranker is None


def test_kb_name_is_validated():
    with pytest.raises(ValidationError):
        KBConfig.model_validate(dict(MINIMAL, kb="Finance Docs"))


def test_beam_bounds_are_validated():
    with pytest.raises(ValidationError):
        KBConfig.model_validate(dict(MINIMAL, retrieval={"beam": {"min_beam": 40, "max_beam": 30}}))


def test_from_yaml(tmp_path):
    path = tmp_path / "finance-docs.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))
    assert KBConfig.from_yaml(path).kb == "finance-docs"


def test_example_config_loads():
    kb = KBConfig.from_yaml(ROOT / "configs" / "example.yaml")
    assert kb.kb == "example"
    assert kb.ner.params["labels"] == ["PERSON", "ORG", "PRODUCT", "DATE"]


def test_settings_from_env_ignores_empty_values():
    settings = Settings.from_env({
        "MAYAKB_DATABASE_URL": "postgresql://u:p@db:5432/mayakb",
        "MAYAKB_OLLAMA_URL": "",
        "MAYAKB_DATA_DIR": "/data",
    })
    assert settings.database_url == "postgresql://u:p@db:5432/mayakb"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.data_dir == Path("/data")
    assert settings.kb_config_path("x") == Path("configs/x.yaml")
