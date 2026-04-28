"""Configuration objects for the RAG subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass(frozen=True)
class RetrieverStoreConfig:
    """Path configuration for one FAISS-backed retriever."""

    name: str
    index_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class RAGConfig:
    """Runtime configuration for the RAG pipeline."""

    embedding_service_url: str
    ollama_url: str
    model: str
    embedding_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 120.0
    retry_attempts: int = 3
    retry_base_seconds: float = 0.5
    default_top_k: int = 5


def build_rag_config() -> RAGConfig:
    """Build RAG config from environment-backed application settings."""
    return RAGConfig(
        embedding_service_url=settings.embedding_service_url,
        ollama_url=settings.ollama_url,
        model=settings.model,
    )


def build_retriever_stores(base_dir: Path) -> dict[str, RetrieverStoreConfig]:
    """Create retriever store map for all supported legal corpora."""
    db_names = ("act", "rules", "book", "circular", "notification", "judgement")
    return {
        name: RetrieverStoreConfig(
            name=name,
            index_path=base_dir / name / "index.faiss",
            metadata_path=base_dir / name / "metadata.json",
        )
        for name in db_names
    }
