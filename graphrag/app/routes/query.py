"""Query API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_query_pipeline
from app.pipelines.query_pipeline import QueryPipeline

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    cypher: str
    records: list[dict]


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest, pipeline: QueryPipeline = Depends(get_query_pipeline)) -> QueryResponse:
    """Answer a user question via GraphRAG pipeline.

    :param request: Request body.
    :param pipeline: Query pipeline dependency.
    :return: GraphRAG answer payload.
    """

    try:
        result = await pipeline.answer_question(request.question)
        return QueryResponse(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
