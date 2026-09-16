"""Retry with exponential backoff for calls to network services."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx


def is_transient(exc: BaseException) -> bool:
    """Connection problems, timeouts, 429 and 5xx are worth retrying; other errors are not."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def with_retries[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[BaseException], bool] = is_transient,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not retryable(exc):
                raise
            sleep(base_delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")
