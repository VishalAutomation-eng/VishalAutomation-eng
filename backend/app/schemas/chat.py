from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request payload for chat streaming endpoint."""

    query: str
    top_k: int = 5
    selected_databases: list[str] | None = None
    document_ids: list[str] | None = None
    filters: dict | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
