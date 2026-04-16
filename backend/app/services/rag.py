import json
from typing import AsyncGenerator

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, DocumentChunk
from app.services.embeddings import embed_texts


def extract_chunks_from_pdf(pdf_bytes: bytes, metadata: dict) -> list[dict]:
    reader = PdfReader(stream=pdf_bytes)
    pages = [p.extract_text() or '' for p in reader.pages]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks: list[dict] = []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_text in splitter.split_text(page_text):
            chunks.append({'text': chunk_text, 'metadata': {**metadata, 'page': page_number}})
    return chunks


async def index_document(db: Session, document: Document, pdf_bytes: bytes) -> None:
    chunk_rows = extract_chunks_from_pdf(pdf_bytes, {'document_id': str(document.id), 'filename': document.name})
    embeddings = await embed_texts([row['text'] for row in chunk_rows])

    for idx, row in enumerate(chunk_rows):
        db.add(
            DocumentChunk(
                document_id=document.id,
                owner_id=document.owner_id,
                chunk_text=row['text'],
                chunk_index=idx,
                metadata_json=row['metadata'],
                embedding=embeddings[idx],
            )
        )
    db.commit()


async def retrieve_context(db: Session, owner_id: str, query: str, top_k: int, document_ids: list[str] | None, filters: dict | None):
    query_embedding = (await embed_texts([query]))[0]
    params = {'owner_id': owner_id, 'embedding': query_embedding, 'top_k': top_k}

    where_parts = ['owner_id = :owner_id']
    if document_ids:
        where_parts.append('document_id = ANY(:doc_ids)')
        params['doc_ids'] = document_ids
    if filters:
        for key, val in filters.items():
            where_parts.append(f"metadata_json ->> '{key}' = :{key}")
            params[key] = str(val)

    sql = text(
        f"""
        SELECT chunk_text, metadata_json,
               embedding <=> CAST(:embedding AS vector) AS distance
        FROM document_chunks
        WHERE {' AND '.join(where_parts)}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )
    return db.execute(sql, params).mappings().all()


async def stream_answer(query: str, context: str) -> AsyncGenerator[str, None]:
    prompt = f"""
You are an assistant that answers only from the given context.
Context:
{context}
Question: {query}
"""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream('POST', settings.ollama_url, json={'model': settings.model, 'prompt': prompt, 'stream': True}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                token = payload.get('response', '')
                if token:
                    yield token
