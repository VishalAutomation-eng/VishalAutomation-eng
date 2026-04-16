from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Document, User
from app.db.session import get_db
from app.schemas.chat import UploadResponse
from app.services.rag import index_document
from app.services.storage import upload_bytes

router = APIRouter(prefix='/documents', tags=['documents'])


@router.post('/upload', response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form('general'),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    content = await file.read()
    s3_key = upload_bytes(content, file.filename, str(current_user.id))

    doc = Document(owner_id=current_user.id, name=file.filename, s3_key=s3_key, metadata_json={'category': category})
    db.add(doc)
    db.commit()
    db.refresh(doc)

    await index_document(db, doc, content)
    return UploadResponse(document_id=str(doc.id), filename=file.filename, status='indexed')


@router.get('')
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    return [{'id': str(d.id), 'name': d.name, 'metadata': d.metadata_json} for d in docs]
