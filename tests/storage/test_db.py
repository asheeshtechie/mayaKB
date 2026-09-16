import numpy as np
import psycopg
import pytest

from mayakb.storage.db import DatabaseUnavailable, apply_migrations, make_pool, pgvector_version, redact

UNREACHABLE = "postgresql://mayakb:s3cret@127.0.0.1:1/mayakb"


def test_redact_hides_password():
    assert redact("postgresql://mayakb:s3cret@db:5432/mayakb") == "postgresql://mayakb:***@db:5432/mayakb"
    assert redact("postgresql://db/mayakb") == "postgresql://db/mayakb"


def test_unreachable_database_error_names_the_setting():
    with pytest.raises(DatabaseUnavailable) as err:
        apply_migrations(UNREACHABLE)
    assert "MAYAKB_DATABASE_URL" in str(err.value)
    assert "s3cret" not in str(err.value)


def test_pool_to_unreachable_database_fails_fast():
    with pytest.raises(DatabaseUnavailable, match="MAYAKB_DATABASE_URL"):
        make_pool(UNREACHABLE, timeout=1)


@pytest.mark.postgres
def test_migrations_apply_once(pg_url):
    assert apply_migrations(pg_url) == ["001_init"]
    assert apply_migrations(pg_url) == []
    with psycopg.connect(pg_url) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )}
        assert {"jobs", "indexes", "schema_migrations"} <= tables
        assert pgvector_version(conn) is not None


@pytest.mark.postgres
def test_pool_registers_vector_type(pg_pool):
    with pg_pool.connection() as conn:
        distance = conn.execute("SELECT %s <=> '[1,0,0]'::vector", (np.array([1.0, 0.0, 0.0]),)).fetchone()[0]
    assert distance == 0.0
