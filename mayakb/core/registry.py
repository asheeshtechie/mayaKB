"""Name -> implementation lookup for every pluggable interface.

Built-in implementations are imported only when first requested, so heavy dependencies
(spaCy, PyTorch) load only when a knowledge base actually uses them.
"""

from __future__ import annotations

import importlib
from typing import Any

KINDS = ("connector", "parser", "chunker", "ner", "embedder", "reranker")

_BUILTINS: dict[tuple[str, str], str] = {
    ("connector", "local"): "mayakb.ingest.connectors.local:LocalConnector",
    ("connector", "s3"): "mayakb.ingest.connectors.s3:S3Connector",
    ("parser", "local"): "mayakb.ingest.parsers.local:LocalParser",
    ("parser", "llamaparse"): "mayakb.ingest.parsers.llamaparse:LlamaParseParser",
    ("chunker", "fixed"): "mayakb.ingest.chunking.fixed:FixedChunker",
    ("chunker", "by_section"): "mayakb.ingest.chunking.by_section:BySectionChunker",
    ("ner", "spacy"): "mayakb.ingest.ner.spacy_ner:SpacyExtractor",
    ("embedder", "ollama"): "mayakb.ingest.embedders.ollama:OllamaEmbedder",
    ("reranker", "cross_encoder"): "mayakb.retrieval.rerankers.cross_encoder:CrossEncoderReranker",
}

_LOADED: dict[tuple[str, str], type] = {}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def register(kind: str, name: str):
    """Class decorator: make an implementation available under `name`."""
    _check_kind(kind)

    def decorator(cls: type) -> type:
        cls.name = name
        _LOADED[(kind, name)] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    _check_kind(kind)
    if (kind, name) in _LOADED:
        return _LOADED[(kind, name)]
    target = _BUILTINS.get((kind, name))
    if target is None:
        raise KeyError(f"no {kind} named {name!r}; available: {available(kind)}")
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    _LOADED[(kind, name)] = cls
    return cls


def create(kind: str, name: str, **params: Any) -> Any:
    return get(kind, name)(**params)


def available(kind: str) -> list[str]:
    _check_kind(kind)
    names = {n for (k, n) in _BUILTINS if k == kind} | {n for (k, n) in _LOADED if k == kind}
    return sorted(names)


def unregister(kind: str, name: str) -> None:
    """For tests: forget a registration."""
    _LOADED.pop((kind, name), None)
