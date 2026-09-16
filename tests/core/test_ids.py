from mayakb.core.ids import canonical_json, content_id, stable_id


def test_content_id_is_sha256_hex():
    assert content_id(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_stable_id_is_deterministic_and_order_sensitive():
    assert stable_id("chunk", {"x": 1}, ["a"]) == stable_id("chunk", {"x": 1}, ["a"])
    assert stable_id("a", "b") != stable_id("b", "a")


def test_stable_id_keeps_parts_separate():
    assert stable_id("ab", "c") != stable_id("a", "bc")


def test_stable_id_length():
    assert len(stable_id("x")) == 32
    assert len(stable_id("x", length=12)) == 12
