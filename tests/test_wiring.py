import warnings

import pytest
import yaml

from mayakb import wiring
from mayakb.core.config import Settings
from mayakb.ingest.runner import ingest
from mayakb.storage.db import DatabaseUnavailable
from mayakb.storage.search.postgres import PostgresSearchBackend
from tests.fakes import make_kb


def write_config(folder, **sections):
    kb = make_kb(**sections)
    (folder / f"{kb.kb}.yaml").write_text(yaml.safe_dump(kb.model_dump(mode="json")))
    return kb


def test_load_kb_errors(tmp_path):
    settings = Settings(configs_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no config"):
        wiring.load_kb("test-kb", settings)
    write_config(tmp_path)
    (tmp_path / "other.yaml").write_text((tmp_path / "test-kb.yaml").read_text())
    with pytest.raises(ValueError, match="expected 'other'"):
        wiring.load_kb("other", settings)
    assert wiring.load_kb("test-kb", settings).kb == "test-kb"


def test_build_runtime_needs_postgres_config_and_url(tmp_path):
    write_config(tmp_path)  # make_kb uses the memory backend
    with pytest.raises(ValueError, match="not available"):
        wiring.build_runtime("test-kb", Settings(configs_dir=tmp_path))
    write_config(tmp_path, index={"backend": "postgres"})
    with pytest.raises(DatabaseUnavailable, match="MAYAKB_DATABASE_URL is not set"):
        wiring.build_runtime("test-kb", Settings(configs_dir=tmp_path))


@pytest.mark.postgres
def test_end_to_end_on_postgres(tmp_path, pg_url):
    from mayakb.storage.db import apply_migrations

    apply_migrations(pg_url)
    configs = tmp_path / "configs"
    configs.mkdir()
    write_config(
        configs,
        docs={"mem://docs/refunds.md": "# Refunds\n\nRefunds are issued within 30 days of the return."},
        index={"backend": "postgres"},
        retrieval={"reranker": {"reranker": "overlap"}},
    )
    settings = Settings(database_url=pg_url, configs_dir=configs, data_dir=tmp_path / "data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runtime = wiring.build_runtime("test-kb", settings)
    try:
        assert isinstance(runtime.backend, PostgresSearchBackend)
        assert ingest(runtime.context()).counts()["succeeded"] == 1
        assert ingest(runtime.context()).counts()["skipped"] == 1
        results = runtime.searcher().search("how long do refunds take")
        assert results.chunks[0].citation.source_uri == "mem://docs/refunds.md"
    finally:
        runtime.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = wiring.search("refunds", "test-kb", settings=settings)
    assert again.chunks
