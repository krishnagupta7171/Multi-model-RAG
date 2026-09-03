from functools import lru_cache

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

    chunk_size: int = Field(
        default=512,
        ge=100,
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
    )

 


@lru_cache()
def get_settings() -> Settings:
    return Settings()