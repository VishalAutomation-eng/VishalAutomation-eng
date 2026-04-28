"""Embedding service client with retries and timeout handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for obtaining embeddings from the external embedding service."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed input texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: If embedding service fails after retries.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        payload = {"texts": texts}
        endpoint = f"{self._config.embedding_service_url.rstrip('/')}/embed"

        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                logger.info(
                    "embedding.request",
                    extra={"endpoint": endpoint, "attempt": attempt, "text_count": len(texts)},
                )
                async with httpx.AsyncClient(timeout=self._config.embedding_timeout_seconds) as client:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                embeddings = data.get("embeddings")
                if not isinstance(embeddings, list):
                    raise RuntimeError("embedding service returned invalid payload")
                return embeddings
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                logger.exception(
                    "embedding.request_failed",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                if attempt >= self._config.retry_attempts:
                    raise RuntimeError("embedding request failed after retries") from exc
                await asyncio.sleep(self._config.retry_base_seconds * (2 ** (attempt - 1)))

        raise RuntimeError("embedding request failed")
