"""LLM service abstraction using Ollama HTTP API."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Client for LLM calls.

    :ivar _client: Async HTTP client.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)

    async def close(self) -> None:
        """Close underlying HTTP client."""

        await self._client.aclose()

    async def generate(self, prompt: str, temperature: float = 0.0, response_format: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response from the configured Ollama model.

        :param prompt: Full prompt input.
        :param temperature: Sampling temperature.
        :param response_format: Optional response format payload.
        :return: Model text response.
        :raises RuntimeError: If request fails after retries.
        """

        payload: Dict[str, Any] = {
            "model": settings.ollama_chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if response_format is not None:
            payload["format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                backoff = min(2 ** (attempt - 1), 8)
                logger.warning("LLM call failed on attempt %s/%s: %s", attempt, settings.llm_max_retries, exc)
                await asyncio.sleep(backoff)

        raise RuntimeError(f"LLM generation failed after retries: {last_error}")

    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate and parse JSON response.

        :param prompt: Prompt expecting JSON output.
        :return: Parsed JSON object.
        :raises ValueError: If model returns invalid JSON.
        """

        content = await self.generate(prompt=prompt, temperature=0.0, response_format={"type": "json_object"})
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON returned by LLM: %s", content)
            raise ValueError("LLM returned invalid JSON") from exc
