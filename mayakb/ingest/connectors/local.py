"""Files under a folder, e.g. one mounted into the app container at /sources."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from mayakb.core.contracts import RawDocument, SourceRef
from mayakb.core.ids import content_id
from mayakb.ingest.connectors.base import DEFAULT_SUFFIXES, Connector, guess_mime


class LocalConnector(Connector):
    name = "local"
    version = "1"

    def __init__(self, path: str, suffixes: list[str] | None = None):
        self.root = Path(path)
        self.suffixes = tuple(s.lower() for s in (suffixes or DEFAULT_SUFFIXES))

    def list(self) -> Iterator[SourceRef]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"source folder not found: {self.root}")
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            stat = path.stat()
            yield SourceRef(
                uri=path.resolve().as_uri(),
                etag=f"{stat.st_size}-{stat.st_mtime_ns}",
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                size=stat.st_size,
            )

    def fetch(self, ref: SourceRef) -> tuple[RawDocument, bytes]:
        path = Path(url2pathname(urlparse(ref.uri).path))
        data = path.read_bytes()
        raw = RawDocument(
            id=content_id(data),
            source_uri=ref.uri,
            connector=self.name,
            mime=guess_mime(path.name),
            etag=ref.etag,
            modified_at=ref.modified_at,
            metadata={"size": len(data), "filename": path.name},
        )
        return raw, data
