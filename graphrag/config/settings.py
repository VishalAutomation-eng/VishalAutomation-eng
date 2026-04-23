"""Application settings for the GraphRAG service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration container for all runtime settings.

    :ivar app_name: FastAPI app name.
    :ivar app_env: Runtime environment name.
    :ivar app_host: Service host.
    :ivar app_port: Service port.
    :ivar log_level: Logging level.
    :ivar neo4j_uri: Neo4j bolt URI.
    :ivar neo4j_user: Neo4j username.
    :ivar neo4j_password: Neo4j password.
    :ivar neo4j_database: Neo4j database name.
    :ivar ollama_base_url: Ollama API base URL.
    :ivar ollama_chat_model: Model name used for chat/completions.
    :ivar ollama_timeout_seconds: Request timeout in seconds.
    :ivar embedding_model_name: Sentence-transformers model for embeddings.
    :ivar chunk_size: Chunk size in characters.
    :ivar chunk_overlap: Chunk overlap in characters.
    :ivar llm_max_retries: Max retries for LLM calls.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "GraphRAG API"
    app_env: str = Field(default="dev")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8001)
    log_level: str = Field(default="INFO")

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="neo4j")
    neo4j_database: str = Field(default="neo4j")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="llama3.1")
    ollama_timeout_seconds: float = Field(default=90.0)

    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=150)

    llm_max_retries: int = Field(default=3)


settings = Settings()
