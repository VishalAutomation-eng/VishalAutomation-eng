"""Client for generating answers from Ollama."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

import httpx

from app.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for the Ollama generate endpoint."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            "You are a helpful legal assistant. Use only the provided context. "
            "If context is insufficient, explicitly say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

    async def generate(self, query: str, context: str, stream: bool = True) -> AsyncGenerator[str, None]:
        """Stream generated answer tokens."""
        payload = {
            "model": self._config.model,
            "prompt": self._build_prompt(query=query, context=context),
            "stream": stream,
        }

        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                logger.info("llm.request", extra={"attempt": attempt, "stream": stream})
                async with httpx.AsyncClient(timeout=self._config.llm_timeout_seconds) as client:
                    async with client.stream("POST", self._config.ollama_url, json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            chunk: dict[str, Any] = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                return
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                logger.exception("llm.request_failed", extra={"attempt": attempt, "error": str(exc)})
                if attempt >= self._config.retry_attempts:
                    raise RuntimeError("llm request failed after retries") from exc
                await asyncio.sleep(self._config.retry_base_seconds * (2 ** (attempt - 1)))
