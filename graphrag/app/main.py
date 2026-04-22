"""FastAPI entry point for GraphRAG service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.pipelines.query_pipeline import QueryPipeline
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router
from app.services.extraction_service import ExtractionService
from app.services.graph_service import GraphService
from app.services.llm_service import LLMService
from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown dependencies.

    :param app: FastAPI app instance.
    """

    llm_service = LLMService()
    graph_service = GraphService()
    extraction_service = ExtractionService(llm_service=llm_service)
    ingestion_pipeline = IngestionPipeline(extraction_service=extraction_service, graph_service=graph_service)
    query_pipeline = QueryPipeline(
        llm_service=llm_service,
        graph_service=graph_service,
        embedding_model_name=settings.embedding_model_name,
    )

    await graph_service.ensure_constraints()

    app.state.llm_service = llm_service
    app.state.graph_service = graph_service
    app.state.ingestion_pipeline = ingestion_pipeline
    app.state.query_pipeline = query_pipeline

    yield

    await llm_service.close()
    await graph_service.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe endpoint.

    :return: Health response.
    """

    return {"status": "ok"}
