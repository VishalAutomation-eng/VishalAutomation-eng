"""Tests for ingestion pipeline."""

import asyncio

from app.pipelines.ingestion_pipeline import IngestionPipeline


class DummyExtractionService:
    """Mock extraction service."""

    async def extract(self, text: str):
        return {
            "entities": [{"name": "Alice", "type": "Person"}],
            "relations": [{"source": "Alice", "target": "Acme", "type": "WORKS_AT"}],
        }


class DummyGraphService:
    """Mock graph service."""

    async def upsert_extraction(self, chunk_id: str, chunk_text: str, extraction):
        return {"chunks": 1, "entities": len(extraction["entities"]), "relations": len(extraction["relations"])}


def test_ingest_text_aggregates_stats() -> None:
    pipeline = IngestionPipeline(DummyExtractionService(), DummyGraphService())
    stats = asyncio.run(pipeline.ingest_text("Alice works at Acme. " * 200))
    assert stats["chunks"] > 0
    assert stats["entities"] == stats["chunks"]
    assert stats["relations"] == stats["chunks"]
