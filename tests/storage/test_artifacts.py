import numpy as np
import pytest

from mayakb.core.contracts import RawDocument
from mayakb.storage import artifacts
from mayakb.storage.artifacts import ArtifactNotFound, FileArtifactStore


@pytest.fixture
def store(tmp_path):
    return FileArtifactStore(tmp_path)


def test_model_round_trip_and_layout(store, tmp_path):
    raw = RawDocument(id="abc", source_uri="file:///a.txt", connector="local", mime="text/plain")
    store.write_model("kb1", "raw", "abc", raw)
    assert (tmp_path / "kb1" / "raw" / "abc.json").is_file()
    assert store.exists("kb1", "raw", "abc")
    assert store.read_model("kb1", "raw", "abc", RawDocument) == raw


def test_bytes_and_json(store):
    store.write_bytes("kb1", "raw", "abc", b"hello")
    assert store.read_bytes("kb1", "raw", "abc") == b"hello"
    store.write_json("kb1", "source_map", "k", {"etag": "1"})
    assert store.read_json("kb1", "source_map", "k") == {"etag": "1"}


def test_missing_artifacts(store):
    with pytest.raises(ArtifactNotFound):
        store.read_bytes("kb1", "raw", "nope")
    assert store.read_json("kb1", "source_map", "nope") is None
    assert not store.exists("kb1", "raw", "nope")


def test_array_round_trip(store):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    store.write_array("kb1", "vectors", "v1", array)
    loaded = store.read_array("kb1", "vectors", "v1")
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, array)


def test_overwrite_replaces_content(store):
    store.write_bytes("kb1", "raw", "abc", b"one")
    store.write_bytes("kb1", "raw", "abc", b"two")
    assert store.read_bytes("kb1", "raw", "abc") == b"two"


@pytest.mark.parametrize("kind,id", [("bogus", "x"), ("raw", "../x"), ("raw", ""), ("raw", ".hidden")])
def test_rejects_bad_kind_or_id(store, kind, id):
    with pytest.raises(ValueError):
        store.write_bytes("kb1", kind, id, b"x")


def test_failed_write_leaves_no_partial_file(store, tmp_path, monkeypatch):
    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.os, "replace", broken_replace)
    with pytest.raises(OSError):
        store.write_bytes("kb1", "raw", "abc", b"data")
    folder = tmp_path / "kb1" / "raw"
    assert not (folder / "abc.bin").exists()
    assert list(folder.iterdir()) == []
