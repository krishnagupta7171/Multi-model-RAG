import time
from contextlib import asynccontextmanager
from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health, ingest, query
from src.observability.logging import get_logger, setup_logging
from src.utils.config import get_settings
from src.utils.exceptions import RAGException

setup_logging()
logger = get_logger(__name__)


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Starting RAG application...")
    logger.info(f"Environment: {getattr(settings, 'environment', 'dev')}")
    logger.info(f"Vector DB: {getattr(settings, 'vector_db_type', 'chroma')}")

    yield

    logger.info("Shutting down RAG application...")

app = FastAPI(title="Multi-Modal RAG Agent API",
    description="Production-grade RAG system with agentic workflows",
    version="0.1.0",lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(CORSMiddleware,
    allow_origins=getattr(settings, "cors_origins", ["*"]),
    allow_credentials=True,allow_methods=["*"],allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    
    request_id = request.headers.get("X-Request-ID", "unknown")

    logger.info(f"Request [{request_id}]: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    response = await call_next(request)

    logger.info(f"Response [{request_id}]: {response.status_code}",
        extra={"request_id": request_id},
    )

    return response

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    logger.error(f"RAG exception: {exc.message}", exc_info=getattr(exc, "original_error", None))

    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict() if hasattr(exc, "to_dict") else {"error": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"Value error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": "ValueError",
            "message": str(exc),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )


app.include_router(health.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "name": "Multi-Modal RAG Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

@app.get("/metrics", tags=["Observability"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1")
async def api_info():
    return {
        "version": "v1",
        "endpoints": {
            "query": "/api/v1/query",
            "ingest": "/api/v1/ingest",
            "health": "/api/v1/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=getattr(settings, "api_host", "0.0.0.0"),
        port=getattr(settings, "api_port", 8000),
        reload=getattr(settings, "is_development", True),
        log_level=getattr(settings, "log_level", "info").lower(),
    )