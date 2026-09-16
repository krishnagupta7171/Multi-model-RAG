import uuid
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from ..generation.LLMGenerator import LLMGenerator
from ..observability.logging import get_logger
from ..retrieval.vector_store import VectorStore
from ..utils.cache import CacheManager, get_cache_manager
from ..utils.config import Settings, get_settings

logger = get_logger(__name__)

# Cached global singleton references
_vector_store: Optional[VectorStore] = None
_llm_generator: Optional[LLMGenerator] = None


def get_settings_dependency() -> Settings:
    
    return get_settings()


def get_cache_dependency() -> CacheManager:
    return get_cache_manager()


def get_llm_generator_dependency(settings: Settings = Depends(get_settings_dependency),) -> LLMGenerator:
    
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMGenerator()
    return _llm_generator


def get_vector_store_dependency(settings: Settings = Depends(get_settings_dependency),) -> VectorStore:
    
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_request_id(x_request_id: Optional[str] = Header(default=None)) -> str:
    
    return x_request_id or str(uuid.uuid4())


async def verify_api_key(x_api_key: Optional[str] = Header(default=None),settings: Settings = Depends(get_settings_dependency),) -> str:
    
    is_prod = getattr(settings, "is_production", False) or getattr(settings, "environment", "dev") == "production"

    if is_prod:
        configured_key = getattr(settings, "api_key", None)
        if not x_api_key or (configured_key and x_api_key != configured_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        logger.debug("API key authenticated successfully")

    return x_api_key or "development"