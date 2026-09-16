import httpx
import pytest

from mayakb.core.retry import is_transient, with_retries


def status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://x")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(code, request=request))


def test_retries_transient_errors_then_succeeds():
    calls, delays = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("down")
        return "ok"

    assert with_retries(flaky, sleep=delays.append) == "ok"
    assert delays == [1.0, 2.0]


def test_gives_up_after_attempts():
    delays = []

    def always_down():
        raise status_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        with_retries(always_down, attempts=3, sleep=delays.append)
    assert len(delays) == 2


def test_does_not_retry_client_errors():
    delays = []

    def bad_request():
        raise status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        with_retries(bad_request, sleep=delays.append)
    assert delays == []


def test_is_transient():
    assert is_transient(status_error(429))
    assert is_transient(httpx.ReadTimeout("slow"))
    assert not is_transient(status_error(404))
    assert not is_transient(ValueError("x"))
