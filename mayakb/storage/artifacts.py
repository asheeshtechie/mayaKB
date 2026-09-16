"""ArtifactStore: job outputs on disk at <root>/<kb>/<kind>/<id>.<ext>, written atomically."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)

KINDS = ("raw", "doc_model", "chunks", "entities", "vectors", "source_map")


class ArtifactNotFound(FileNotFoundError):
    pass


class ArtifactStore(Protocol):
    def exists(self, kb: str, kind: str, id: str, ext: str = "json") -> bool: ...
    def write_bytes(self, kb: str, kind: str, id: str, data: bytes, ext: str = "bin") -> None: ...
    def read_bytes(self, kb: str, kind: str, id: str, ext: str = "bin") -> bytes: ...
    def write_model(self, kb: str, kind: str, id: str, model: BaseModel) -> None: ...
    def read_model(self, kb: str, kind: str, id: str, cls: type[M]) -> M: ...
    def write_json(self, kb: str, kind: str, id: str, value: Any) -> None: ...
    def read_json(self, kb: str, kind: str, id: str) -> Any | None: ...
    def write_array(self, kb: str, kind: str, id: str, array: np.ndarray) -> None: ...
    def read_array(self, kb: str, kind: str, id: str) -> np.ndarray: ...


def atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


class FileArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, kb: str, kind: str, id: str, ext: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown artifact kind {kind!r}")
        if not id or "/" in id or "\\" in id or id.startswith("."):
            raise ValueError(f"invalid artifact id {id!r}")
        return self.root / kb / kind / f"{id}.{ext}"

    def exists(self, kb: str, kind: str, id: str, ext: str = "json") -> bool:
        return self.path(kb, kind, id, ext).is_file()

    def write_bytes(self, kb: str, kind: str, id: str, data: bytes, ext: str = "bin") -> None:
        atomic_write(self.path(kb, kind, id, ext), data)

    def read_bytes(self, kb: str, kind: str, id: str, ext: str = "bin") -> bytes:
        path = self.path(kb, kind, id, ext)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise ArtifactNotFound(str(path)) from None

    def write_model(self, kb: str, kind: str, id: str, model: BaseModel) -> None:
        self.write_bytes(kb, kind, id, model.model_dump_json().encode("utf-8"), ext="json")

    def read_model(self, kb: str, kind: str, id: str, cls: type[M]) -> M:
        return cls.model_validate_json(self.read_bytes(kb, kind, id, ext="json"))

    def write_json(self, kb: str, kind: str, id: str, value: Any) -> None:
        data = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        self.write_bytes(kb, kind, id, data, ext="json")

    def read_json(self, kb: str, kind: str, id: str) -> Any | None:
        try:
            return json.loads(self.read_bytes(kb, kind, id, ext="json"))
        except ArtifactNotFound:
            return None

    def write_array(self, kb: str, kind: str, id: str, array: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, array, allow_pickle=False)
        self.write_bytes(kb, kind, id, buf.getvalue(), ext="npy")

    def read_array(self, kb: str, kind: str, id: str) -> np.ndarray:
        return np.load(io.BytesIO(self.read_bytes(kb, kind, id, ext="npy")), allow_pickle=False)
