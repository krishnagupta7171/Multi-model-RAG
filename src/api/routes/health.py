from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from ...observability.logging import get_logger
from ...utils.cache import CacheManager
from ...utils.config import Settings
from ..deps import get_cache_dependency, get_settings_dependency
from ..payload import HealthResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",response_model=HealthResponse,status_code=status.HTTP_200_OK,summary="Comprehensive health check",)
async def health_check(settings: Settings = Depends(get_settings_dependency),cache: CacheManager = Depends(get_cache_dependency),) -> JSONResponse:
    
    dependencies = {
        "config": "healthy",
        "cache": "healthy" if getattr(cache, "is_available", True) else "degraded",
        "vector_store": "healthy",
    }

    # Determine aggregated status
    is_healthy = all(v == "healthy" for v in dependencies.values())
    overall_status = "healthy" if is_healthy else "degraded"
    version = getattr(settings, "version", "0.1.0")

    response_data = HealthResponse(status=overall_status,version=version,dependencies=dependencies,)

    return JSONResponse(status_code=status.HTTP_200_OK if is_healthy else status.HTTP_207_MULTI_STATUS,content=response_data.model_dump(),)


@router.get("/live",status_code=status.HTTP_200_OK,summary="Liveness probe",)
async def liveness() -> dict:
    return {"status": "alive"}