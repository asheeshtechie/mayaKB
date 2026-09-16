"""Knowledge base config (one YAML file per KB) and runtime settings (environment variables)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_KB_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

# YAML section -> the key inside it that names the implementation
_SECTION_KEYS = {
    "source": "connector",
    "parse": "parser",
    "chunk": "strategy",
    "ner": "extractor",
    "embed": "embedder",
    "index": "backend",
}


class ImplConfig(BaseModel):
    """Which implementation to use and the parameters to construct it with."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _to_impl(section: Any, key: str) -> Any:
    """{connector: s3, bucket: x} -> {name: s3, params: {bucket: x}}. Other shapes pass through."""
    if not isinstance(section, Mapping) or key not in section:
        return section
    return {"name": section[key], "params": {k: v for k, v in section.items() if k != key}}


class BeamConfig(BaseModel):
    policy: Literal["adaptive", "fixed"] = "adaptive"
    depth: int = Field(50, ge=1)  # hits per retriever
    size: int = Field(30, ge=1)  # fixed policy only
    min_beam: int = Field(5, ge=1)
    max_beam: int = Field(30, ge=1)
    agree_top: int = Field(20, ge=1)
    dense_rel: float = Field(0.9, ge=0, le=1)
    keyword_rel: float = Field(0.6, ge=0, le=1)
    fallback_widen: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> BeamConfig:
        if self.min_beam > self.max_beam:
            raise ValueError("min_beam must be <= max_beam")
        return self


class CutConfig(BaseModel):
    min_k: int = Field(2, ge=1)
    max_k: int = Field(10, ge=1)
    min_score: float = Field(0.3, ge=0, le=1)
    relative_to_top: float = Field(0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> CutConfig:
        if self.min_k > self.max_k:
            raise ValueError("min_k must be <= max_k")
        return self


def _default_reranker() -> ImplConfig:
    return ImplConfig(name="cross_encoder", params={"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})


class RetrievalConfig(BaseModel):
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(0.6, ge=0, le=1)  # weighted fusion only
    token_budget: int = Field(3000, ge=1)
    beam: BeamConfig = Field(default_factory=BeamConfig)
    cut: CutConfig = Field(default_factory=CutConfig)
    reranker: ImplConfig | None = Field(default_factory=_default_reranker)

    @field_validator("reranker", mode="before")
    @classmethod
    def _reranker_section(cls, value: Any) -> Any:
        return _to_impl(value, "reranker")


class KBConfig(BaseModel):
    kb: str
    source: ImplConfig
    parse: ImplConfig
    chunk: ImplConfig
    ner: ImplConfig | None = None
    embed: ImplConfig
    index: ImplConfig = Field(
        default_factory=lambda: ImplConfig(name="postgres", params={"text_search_language": "english"})
    )
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @model_validator(mode="before")
    @classmethod
    def _flatten_sections(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        for section, key in _SECTION_KEYS.items():
            if section in data:
                data[section] = _to_impl(data[section], key)
        return data

    @field_validator("kb")
    @classmethod
    def _kb_name(cls, value: str) -> str:
        if not _KB_NAME.match(value):
            raise ValueError("kb must be lowercase letters, digits, '-' or '_' (max 63 characters)")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> KBConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


class Settings(BaseModel):
    """Deployment settings. Never stored in KB config files."""

    database_url: str | None = None
    ollama_url: str = "http://localhost:11434"
    data_dir: Path = Path("data")
    configs_dir: Path = Path("configs")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        values = {
            "database_url": env.get("MAYAKB_DATABASE_URL"),
            "ollama_url": env.get("MAYAKB_OLLAMA_URL"),
            "data_dir": env.get("MAYAKB_DATA_DIR"),
            "configs_dir": env.get("MAYAKB_CONFIGS_DIR"),
        }
        return cls(**{k: v for k, v in values.items() if v})

    def kb_config_path(self, kb: str) -> Path:
        return self.configs_dir / f"{kb}.yaml"
