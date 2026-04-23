"""Text chunking and normalization utilities."""

import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalize text for stable extraction.

    :param text: Raw user input.
    :return: Normalized text.
    """

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\t ]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks.

    :param text: Input text.
    :param chunk_size: Maximum characters per chunk.
    :param chunk_overlap: Overlap characters between adjacent chunks.
    :return: List of chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    n = len(normalized)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(normalized[start:end])
        if end == n:
            break
        start = end - chunk_overlap

    return chunks
