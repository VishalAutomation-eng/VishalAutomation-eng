"""Tests for extraction service."""

import asyncio

from app.services.extraction_service import ExtractionService


class DummyLLMService:
    """Mock LLM service for extraction tests."""

    def __init__(self) -> None:
        self.last_prompt = ""

    async def generate_json(self, prompt: str):
        self.last_prompt = prompt
        return {"entities": [], "relations": []}


def test_extract_prompt_formats_without_key_error() -> None:
    llm = DummyLLMService()
    svc = ExtractionService(llm)

    result = asyncio.run(svc.extract("Alice works at Acme."))

    assert result == {"entities": [], "relations": []}
    assert '"entities"' in llm.last_prompt
    assert "Alice works at Acme." in llm.last_prompt
