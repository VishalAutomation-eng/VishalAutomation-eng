"""Base interfaces for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.documents import RetrievedDocument


class BaseRetriever(ABC):
    """Abstract interface for all retriever implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retriever / database."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """Retrieve relevant documents for query."""
