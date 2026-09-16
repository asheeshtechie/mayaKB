import os

import pytest

from mayakb.core import registry
from mayakb.core.ids import content_id
from mayakb.ingest.connectors.base import DOCX_MIME, guess_mime
from mayakb.ingest.connectors.local import LocalConnector


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b file.md").write_text("# B")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "d.txt").write_text("hidden")
    return tmp_path


def test_lists_supported_files_only(folder):
    refs = list(LocalConnector(str(folder)).list())
    names = [ref.uri.rsplit("/", 1)[-1] for ref in refs]
    assert names == ["a.txt", "b%20file.md"]
    assert all(ref.uri.startswith("file://") and ref.etag for ref in refs)


def test_fetch_returns_bytes_and_metadata(folder):
    connector = LocalConnector(str(folder))
    ref = next(r for r in connector.list() if r.uri.endswith(".md"))
    raw, data = connector.fetch(ref)
    assert data == b"# B"
    assert raw.id == content_id(b"# B")
    assert raw.mime == "text/markdown"
    assert raw.metadata["filename"] == "b file.md"
    assert raw.connector == "local"


def test_etag_changes_when_file_changes(folder):
    connector = LocalConnector(str(folder))
    before = next(connector.list()).etag
    path = folder / "a.txt"
    path.write_text("alpha, now longer")
    os.utime(path, ns=(1, 2_000_000_000))
    after = next(connector.list()).etag
    assert before != after


def test_missing_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(LocalConnector(str(tmp_path / "missing")).list())


def test_guess_mime():
    assert guess_mime("X.PDF") == "application/pdf"
    assert guess_mime("a.docx") == DOCX_MIME
    assert guess_mime("a.unknownext") == "application/octet-stream"


def test_registered_as_local():
    assert registry.get("connector", "local") is LocalConnector
