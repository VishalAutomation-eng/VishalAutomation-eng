from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.chat import ChatRequest
from app.services.cache import cache_key, get_cached_response, set_cached_response
from app.services.rag import retrieve_context, stream_answer

router = APIRouter(prefix='/chat', tags=['chat'])


@router.post('/stream')
async def chat_stream(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    key = cache_key(str(current_user.id), payload.query, payload.filters)
    cached = get_cached_response(key)

    async def event_generator():
        if cached:
            yield {'event': 'token', 'data': cached}
            yield {'event': 'done', 'data': '[DONE]'}
            return

        rows = await retrieve_context(
            db=db,
            owner_id=str(current_user.id),
            query=payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
            filters=payload.filters,
        )
        context = '\n\n'.join([r['chunk_text'] for r in rows])

        full_answer = ''
        async for token in stream_answer(payload.query, context):
            full_answer += token
            yield {'event': 'token', 'data': token}

        set_cached_response(key, full_answer)
        yield {'event': 'done', 'data': '[DONE]'}

    return EventSourceResponse(event_generator())
