"""FAISS retriever implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.rag.documents import RetrievedDocument
from app.rag.embeddings import EmbeddingClient
from app.rag.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover - environment dependent
    faiss = None


class FaissRetriever(BaseRetriever):
    """Retriever backed by a local FAISS index and metadata file."""

    def __init__(
        self,
        name: str,
        embedding_client: EmbeddingClient,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        self._name = name.lower()
        self._embedding_client = embedding_client
        self._index_path = index_path
        self._metadata_path = metadata_path

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """Retrieve top-k chunks for query from the FAISS store."""
        if faiss is None:
            raise RuntimeError("faiss is not installed; add faiss-cpu to dependencies")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip():
            raise ValueError("query must not be empty")

        if not self._index_path.exists() or not self._metadata_path.exists():
            logger.warning(
                "retriever.store_missing",
                extra={
                    "db": self._name,
                    "index_path": str(self._index_path),
                    "metadata_path": str(self._metadata_path),
                },
            )
            return []

        query_vector = await self._embedding_client.embed_texts([query])
        vector = np.array(query_vector, dtype="float32")

        def _search() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
            index = faiss.read_index(str(self._index_path))
            with self._metadata_path.open("r", encoding="utf-8") as file:
                metadata: list[dict[str, Any]] = json.load(file)
            distances, indices = index.search(vector, top_k)
            return distances, indices, metadata

        distances, indices, metadata = await asyncio.to_thread(_search)

        docs: list[RetrievedDocument] = []
        for score, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(metadata):
                continue
            row = metadata[idx]
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            docs.append(
                RetrievedDocument(
                    text=text,
                    score=float(score),
                    source_db=self._name,
                    metadata={k: v for k, v in row.items() if k != "text"},
                )
            )

        logger.info("retriever.results", extra={"db": self._name, "result_count": len(docs)})
        return docs
