from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=('.env', '../.env'),
        env_file_encoding='utf-8',
        extra='ignore',
    )

    app_name: str = 'PDF RAG Assistant'
    port: int = 8005
    is_production: bool = False

    postgres_user: str = 'data_analysis_user'
    postgres_password: str = 'user123'
    postgres_host: str = 'localhost'
    postgres_port: int = 5432
    postgres_db: str = 'data_analysis'

    redis_url: str = 'redis://localhost:6379/0'

    jwt_secret_key: str = 'change-this-in-production'
    jwt_algorithm: str = 'HS256'
    jwt_exp_minutes: int = 120

    admin_email: str = 'admin@admin.com'
    admin_password: str = 'admin123'

    s3_bucket: str = 'local-dev-bucket'
    aws_region: str = 'us-east-1'
    aws_access_key_id: str = 'local-dev-key'
    aws_secret_access_key: str = 'local-dev-secret'

    embedding_service_url: str = 'http://localhost:8050'
    ollama_url: str = 'http://localhost:11434/api/generate'
    model: str = 'gpt-oss:120b'

    chunk_size: int = 1000
    chunk_overlap: int = 150

    cors_origins: str = 'http://localhost:5173'

    @property
    def database_url(self) -> str:
        return (
            f'postgresql+psycopg://{self.postgres_user}:{self.postgres_password}'
            f'@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}'
        )


settings = Settings()
