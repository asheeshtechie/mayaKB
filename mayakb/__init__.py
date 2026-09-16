"""mayaKB: job-based ingestion and hybrid retrieval."""

__version__ = "0.1.0"


def search(query, kb, **kwargs):
    """Search a knowledge base using settings from the environment. See mayakb.wiring.search."""
    from mayakb.wiring import search as _search

    return _search(query, kb, **kwargs)
