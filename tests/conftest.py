import os

import psycopg
import pytest

from mayakb.storage.db import apply_migrations, make_pool


@pytest.fixture
def pg_url():
    """A clean database. MAYAKB_TEST_DATABASE_URL must point at a THROWAWAY Postgres: it is wiped."""
    url = os.environ.get("MAYAKB_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set MAYAKB_TEST_DATABASE_URL to run Postgres tests")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        conn.execute("CREATE SCHEMA public")
    return url


@pytest.fixture
def pg_pool(pg_url):
    apply_migrations(pg_url)
    pool = make_pool(pg_url, max_size=4)
    yield pool
    pool.close()
