"""Deterministic ids: the same inputs give the same id on every machine and every run."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_id(data: bytes) -> str:
    """sha256 of raw bytes as hex. Used as RawDocument.id."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal dicts serialize identically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(*parts: Any, length: int = 32) -> str:
    """Hash of the parts in order. Strings are used as-is, anything else as canonical JSON."""
    digest = hashlib.sha256()
    for part in parts:
        text = part if isinstance(part, str) else canonical_json(part)
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:length]
