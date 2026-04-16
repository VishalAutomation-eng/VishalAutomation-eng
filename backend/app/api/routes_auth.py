from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail='Invalid credentials')
    token = create_access_token(user.email)
    return TokenResponse(access_token=token)


@router.post('/bootstrap-admin', status_code=201)
def bootstrap_admin(db: Session = Depends(get_db)):
    from app.core.config import settings

    existing = db.query(User).filter(User.email == settings.admin_email).first()
    if existing:
        return {'status': 'already_exists'}

    db.add(User(email=settings.admin_email, hashed_password=hash_password(settings.admin_password)))
    db.commit()
    return {'status': 'created'}
