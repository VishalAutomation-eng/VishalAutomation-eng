"""Old RAG pipeline upgraded with semantic chunking + embedding retrieval."""

from __future__ import annotations

from typing import Sequence

from common import EMBEDDING_API_BASE, EMBEDDING_MODEL, build_vector_index, semantic_chunk_text


def build_old_rag_index(
    pages: Sequence[str],
    embedding_api_base: str = EMBEDDING_API_BASE,
    embedding_model: str = EMBEDDING_MODEL,
) -> object:
    """Build Old RAG index using semantic chunks and embedding search."""
    merged = "\n".join(pages)
    chunks = semantic_chunk_text(
        merged,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
    )

    docs = [
        {"content": chunk, "metadata": {"source": "old_rag_semantic_chunk", "chunk": i + 1}}
        for i, chunk in enumerate(chunks)
    ]
    return build_vector_index(docs, embedding_api_base=embedding_api_base, embedding_model=embedding_model)
