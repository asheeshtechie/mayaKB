import pytest

from mayakb.core.text import count_tokens, normalize_entity, windows, word_pieces, words


def test_count_tokens_counts_whitespace_words():
    assert count_tokens("one two\nthree  ") == 3
    assert count_tokens("") == 0


def test_word_pieces_keep_trailing_whitespace():
    assert word_pieces("a b\n\nc") == ["a ", "b\n\n", "c"]
    assert "".join(word_pieces("  x y")) == "x y"


def test_words_are_lowercase_word_characters():
    assert words("Refunds, within 30-days!") == ["refunds", "within", "30", "days"]


def test_windows_overlap():
    assert windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]


def test_windows_short_and_empty_input():
    assert windows(3, 4, 1) == [(0, 3)]
    assert windows(0, 4, 1) == []


def test_windows_rejects_bad_overlap():
    with pytest.raises(ValueError):
        windows(5, 4, 4)


def test_normalize_entity():
    assert normalize_entity("  Acme   CORP ") == "acme corp"
