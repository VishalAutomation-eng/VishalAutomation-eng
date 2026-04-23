"""Ingestion API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_ingestion_pipeline
from app.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    """Incoming ingestion payload."""

    text: str = Field(..., min_length=1, description="Raw text to ingest")


class IngestResponse(BaseModel):
    """Ingestion response payload."""

    success: bool
    stats: dict[str, int]


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest, pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)) -> IngestResponse:
    """Ingest text into GraphRAG knowledge graph.

    :param request: Input request body.
    :param pipeline: Ingestion pipeline dependency.
    :return: Ingestion status and stats.
    """

    try:
        stats = await pipeline.ingest_text(request.text)
        return IngestResponse(success=True, stats=stats)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
