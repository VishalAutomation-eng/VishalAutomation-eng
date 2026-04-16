from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.routes_auth import router as auth_router
from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as docs_router
from app.core.config import settings
from app.db.session import Base, engine

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(',')],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(auth_router)
app.include_router(docs_router)
app.include_router(chat_router)


def repair_legacy_schema_if_needed() -> None:
    """
    Local-dev safety: older DB snapshots may have a `users` table without `id`.
    That breaks FK creation for `documents.owner_id -> users.id`.
    """
    with engine.begin() as conn:
        try:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        except Exception:
            # extension creation can fail on restricted DB roles; continue
            pass

        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())
        if 'users' not in table_names:
            return

        user_columns = {column['name'] for column in inspector.get_columns('users')}
        if 'id' in user_columns:
            return

        if settings.is_production:
            raise RuntimeError(
                'Legacy users table detected (missing id column). '
                'Run a proper migration before starting in production.'
            )

        conn.execute(text('DROP TABLE IF EXISTS document_chunks CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS documents CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS users CASCADE'))


@app.on_event('startup')
def startup_event():
    repair_legacy_schema_if_needed()
    Base.metadata.create_all(bind=engine)


@app.get('/health')
def health():
    return {'status': 'ok'}
