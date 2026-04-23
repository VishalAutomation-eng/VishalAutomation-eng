"""Ingestion pipeline orchestrating chunking, extraction, and graph writes."""

import asyncio
import logging
import uuid
from typing import Dict

from app.services.extraction_service import ExtractionService
from app.services.graph_service import GraphService
from app.utils.chunking import chunk_text, normalize_text
from config.settings import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end text ingestion pipeline."""

    def __init__(self, extraction_service: ExtractionService, graph_service: GraphService) -> None:
        self._extraction_service = extraction_service
        self._graph_service = graph_service

    async def ingest_text(self, text: str) -> Dict[str, int]:
        """Ingest raw text into the graph.

        :param text: Raw text payload.
        :return: Aggregated graph write stats.
        """

        normalized = normalize_text(text)
        chunks = chunk_text(normalized, settings.chunk_size, settings.chunk_overlap)

        totals = {"chunks": 0, "entities": 0, "relations": 0}
        if not chunks:
            return totals

        for index, chunk in enumerate(chunks):
            chunk_id = f"chunk-{uuid.uuid4()}-{index}"
            extraction = await self._extraction_service.extract(chunk)
            write_stats = await self._graph_service.upsert_extraction(chunk_id=chunk_id, chunk_text=chunk, extraction=extraction)

            totals["chunks"] += write_stats["chunks"]
            totals["entities"] += write_stats["entities"]
            totals["relations"] += write_stats["relations"]

            await asyncio.sleep(0)

        logger.info("Ingestion completed: %s", totals)
        return totals
