"""Page RAG pipeline using page embeddings for retrieval."""

from __future__ import annotations

from typing import Sequence

from common import EMBEDDING_API_BASE, EMBEDDING_MODEL, build_vector_index


def build_page_rag_index(
    pages: Sequence[str],
    embedding_api_base: str = EMBEDDING_API_BASE,
    embedding_model: str = EMBEDDING_MODEL,
) -> object:
    """Build Page RAG embedding index at page granularity."""
    docs = [
        {"content": page_text, "metadata": {"source": "page_rag_embedding", "page": i + 1}}
        for i, page_text in enumerate(pages)
    ]
    return build_vector_index(docs, embedding_api_base=embedding_api_base, embedding_model=embedding_model)
