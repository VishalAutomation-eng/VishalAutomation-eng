"""Dependency providers for FastAPI routes."""

from fastapi import Request

from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.pipelines.query_pipeline import QueryPipeline


async def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Provide ingestion pipeline from app state.

    :param request: FastAPI request.
    :return: Ingestion pipeline.
    """

    return request.app.state.ingestion_pipeline


async def get_query_pipeline(request: Request) -> QueryPipeline:
    """Provide query pipeline from app state.

    :param request: FastAPI request.
    :return: Query pipeline.
    """

    return request.app.state.query_pipeline
