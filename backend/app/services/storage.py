import uuid

import boto3
from botocore.client import Config

from app.core.config import settings


s3_client = boto3.client(
    's3',
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    config=Config(signature_version='s3v4'),
)


def upload_bytes(content: bytes, filename: str, owner_id: str) -> str:
    key = f'pdfs/{owner_id}/{uuid.uuid4()}-{filename}'
    s3_client.put_object(Bucket=settings.s3_bucket, Key=key, Body=content, ContentType='application/pdf')
    return key
