import logging

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.rag.main import pipeline
from app.schemas.chat import ChatRequest
from app.services.cache import cache_key, get_cached_response, set_cached_response

router = APIRouter(prefix='/chat', tags=['chat'])
logger = logging.getLogger(__name__)


@router.post('/stream')
async def chat_stream(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    del db  # Database session is intentionally unused for FAISS-based retrieval.

    key = cache_key(
        str(current_user.id),
        payload.query,
        {
            'filters': payload.filters,
            'selected_databases': payload.selected_databases,
            'top_k': payload.top_k,
        },
    )
    cached = get_cached_response(key)

    async def event_generator():
        if cached:
            yield {'event': 'token', 'data': cached}
            yield {'event': 'done', 'data': '[DONE]'}
            return

        try:
            logger.info(
                'chat.request',
                extra={
                    'user_id': str(current_user.id),
                    'selected_databases': payload.selected_databases,
                    'top_k': payload.top_k,
                },
            )
            full_answer = ''
            async for token in pipeline.answer(
                query=payload.query,
                selected_databases=payload.selected_databases,
                top_k=payload.top_k,
            ):
                full_answer += token
                yield {'event': 'token', 'data': token}

            set_cached_response(key, full_answer)
            yield {'event': 'done', 'data': '[DONE]'}
        except ValueError as exc:
            logger.exception('chat.validation_error', extra={'error': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive runtime catch
            logger.exception('chat.processing_error', extra={'error': str(exc)})
            raise HTTPException(status_code=502, detail='Failed to generate response') from exc

    return EventSourceResponse(event_generator())
