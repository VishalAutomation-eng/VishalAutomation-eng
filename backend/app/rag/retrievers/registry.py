"""Retriever registry and selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.retrievers.base import BaseRetriever


@dataclass
class RetrieverRegistry:
    """Container for named retrievers with validation utilities."""

    retrievers: dict[str, BaseRetriever]

    def select(self, selected_dbs: list[str] | None) -> list[BaseRetriever]:
        """Select retrievers by requested database names.

        Args:
            selected_dbs: Names of requested DBs. If None or empty, all are used.

        Raises:
            ValueError: If any provided DB name is unsupported.
        """
        if not selected_dbs:
            return list(self.retrievers.values())

        normalized = [db.strip().lower() for db in selected_dbs if db and db.strip()]
        unknown = sorted(set(normalized) - set(self.retrievers.keys()))
        if unknown:
            raise ValueError(f"Unsupported databases: {', '.join(unknown)}")

        return [self.retrievers[name] for name in normalized]

    @property
    def supported_databases(self) -> list[str]:
        """Get sorted list of supported database names."""
        return sorted(self.retrievers.keys())
