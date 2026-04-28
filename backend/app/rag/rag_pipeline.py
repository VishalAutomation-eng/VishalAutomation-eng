"""Orchestration layer for retrieval augmented generation."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from app.rag.documents import RetrievedDocument
from app.rag.llm_client import OllamaClient
from app.rag.retrievers.registry import RetrieverRegistry

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinates retrievers and LLM generation."""

    def __init__(self, registry: RetrieverRegistry, llm_client: OllamaClient) -> None:
        self._registry = registry
        self._llm_client = llm_client

    async def retrieve(
        self,
        query: str,
        selected_databases: list[str] | None,
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Retrieve results from selected databases."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        retrievers = self._registry.select(selected_databases)
        logger.info(
            "rag.retrieval_started",
            extra={"selected_databases": [retriever.name for retriever in retrievers], "top_k": top_k},
        )

        result_sets = await asyncio.gather(*(retriever.retrieve(query=query, top_k=top_k) for retriever in retrievers))
        merged = [doc for docs in result_sets for doc in docs]
        merged.sort(key=lambda item: item.score)

        logger.info("rag.retrieval_finished", extra={"result_count": len(merged)})
        return merged

    @staticmethod
    def build_context(documents: list[RetrievedDocument]) -> str:
        """Format retrieved chunks for the generation prompt."""
        if not documents:
            return "No relevant context retrieved from selected databases."

        chunks: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            chunks.append(f"[{idx}] Source={doc.source_db}; Score={doc.score:.4f}\n{doc.text}")
        return "\n\n".join(chunks)

    async def answer(
        self,
        query: str,
        selected_databases: list[str] | None,
        top_k: int,
    ) -> AsyncGenerator[str, None]:
        """Retrieve evidence and stream model answer."""
        docs = await self.retrieve(query=query, selected_databases=selected_databases, top_k=top_k)
        context = self.build_context(docs)
        async for token in self._llm_client.generate(query=query, context=context, stream=True):
            yield token
