from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: list[str] | None = None
    filters: dict | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
