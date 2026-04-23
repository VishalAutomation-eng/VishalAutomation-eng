"""Tests for chunking utilities."""

from app.utils.chunking import chunk_text, normalize_text


def test_normalize_text_removes_extra_whitespace() -> None:
    text = "A\t\tB\r\n\r\n\r\nC"
    assert normalize_text(text) == "A B\n\nC"


def test_chunk_text_overlap() -> None:
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]
