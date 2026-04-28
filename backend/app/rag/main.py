"""Bootstrap helpers for constructing the RAG pipeline via dependency injection."""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings
from app.rag.config import build_rag_config, build_retriever_stores
from app.rag.embeddings import EmbeddingClient
from app.rag.llm_client import OllamaClient
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retrievers.faiss_retriever import FaissRetriever
from app.rag.retrievers.registry import RetrieverRegistry

logger = logging.getLogger(__name__)


def build_pipeline() -> RAGPipeline:
    """Construct a production-ready RAG pipeline instance."""
    rag_config = build_rag_config()
    embeddings = EmbeddingClient(rag_config)
    llm = OllamaClient(rag_config)

    base_store_path = Path(settings.faiss_db_root).resolve()
    stores = build_retriever_stores(base_store_path)

    retrievers = {
        name: FaissRetriever(
            name=cfg.name,
            embedding_client=embeddings,
            index_path=cfg.index_path,
            metadata_path=cfg.metadata_path,
        )
        for name, cfg in stores.items()
    }
    logger.info("rag.pipeline_initialized", extra={"databases": list(retrievers.keys())})
    return RAGPipeline(registry=RetrieverRegistry(retrievers=retrievers), llm_client=llm)


pipeline = build_pipeline()
