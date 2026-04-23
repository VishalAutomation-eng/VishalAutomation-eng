"""Tests for query pipeline."""

import asyncio

from app.pipelines.query_pipeline import QueryPipeline


class DummyLLM:
    """Mock LLM service."""

    async def generate(self, prompt: str, temperature: float = 0.0):
        if "Return ONLY a valid read-only Cypher query" in prompt:
            return "MATCH (p:Entity {type: 'Person'}) RETURN p.name AS name LIMIT 25"
        return "Alice"


class DummyGraph:
    """Mock graph service."""

    async def graph_schema_hint(self) -> str:
        return "Labels: ['Person']; Relationships: ['WORKS_AT']"

    async def run_query(self, cypher_query: str):
        return [{"name": "Alice"}]

    async def fetch_chunks(self, limit: int = 200):
        return []


def test_answer_question_returns_answer() -> None:
    pipeline = QueryPipeline(DummyLLM(), DummyGraph(), "sentence-transformers/all-MiniLM-L6-v2")
    result = asyncio.run(pipeline.answer_question("Who works at Acme?"))
    assert result["answer"] == "Alice"
    assert "MATCH" in result["cypher"]
