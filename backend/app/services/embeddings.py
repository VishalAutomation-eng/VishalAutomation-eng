import httpx

from app.core.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.embedding_service_url.rstrip('/')}/embed",
            json={'texts': texts},
        )
        response.raise_for_status()
        data = response.json()
        return data['embeddings']
