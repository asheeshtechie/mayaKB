import pytest

from mayakb.core import registry


class Dummy:
    version = "1"

    def __init__(self, size: int = 1):
        self.size = size


@pytest.fixture
def dummy():
    registry.register("parser", "dummy")(Dummy)
    yield Dummy
    registry.unregister("parser", "dummy")


def test_register_get_and_create(dummy):
    assert registry.get("parser", "dummy") is dummy
    assert dummy.name == "dummy"
    assert registry.create("parser", "dummy", size=3).size == 3


def test_available_lists_builtins_and_registered(dummy):
    names = registry.available("parser")
    assert {"local", "llamaparse", "dummy"} <= set(names)


def test_unknown_name_lists_what_is_available():
    with pytest.raises(KeyError, match="available"):
        registry.get("chunker", "nope")


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        registry.get("widget", "x")
