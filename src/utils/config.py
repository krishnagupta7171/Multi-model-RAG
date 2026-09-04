from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = Field(default="openai/gpt-oss-120b",description="Groq model to use",)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=4096, ge=1, le=200000)



    # Chunking Configuration

    chunk_size: int = Field(default=512,ge=100,)

    chunk_overlap: int = Field(default=50,ge=0,)


    # Embedding Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2",)

    embedding_dimension: int = Field(default=384,ge=1,)

    # Application / Logging Configuration
    environment: str = Field(default="development",)

    log_level: str = Field(default="INFO",)

    is_production: bool = Field(default=False,)

    # Vector Database Configuration
    vector_db_type: str = Field(default="chroma",description="Vector database type")

    qdrant_url: str = Field(default="http://localhost:6333",description="Qdrant server URL")

    qdrant_api_key: Optional[str] = Field(default=None,description="Qdrant API key")

    chroma_persist_dir: str = Field(default="./data/chroma",description="ChromaDB persistence directory")

 


@lru_cache()
def get_settings() -> Settings:
    return Settings()