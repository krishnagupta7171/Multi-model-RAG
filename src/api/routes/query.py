from fastapi import APIRouter, Depends, HTTPException, status
from src.agents.multimodal_agent import MultimodalRAGAgent
from src.api.deps import (get_cache_dependency,get_request_id,verify_api_key,get_rag_agent_dependency)
from src.api.payload import DocumentResponse, QueryRequest, QueryResponse
from src.observability.logging import get_logger
from src.observability.trace import trace_event
from src.utils.cache import CacheManager

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("",response_model=QueryResponse,status_code=status.HTTP_200_OK,summary="Execute multi-modal RAG query",)


async def query_endpoint(request: QueryRequest,agent: MultimodalRAGAgent = Depends(get_rag_agent_dependency),cache: CacheManager = Depends(get_cache_dependency),
                         request_id: str = Depends(get_request_id),api_key: str = Depends(verify_api_key),) -> QueryResponse:
    trace_event("query_received", {"query": request.query, "request_id": request_id})

    cache_key = f"query:{hash(request.query)}:{request.top_k}:{request.use_reflection}"

    # 1. Cache hit check
    try:
        cached_result = await cache.get(cache_key)
        if cached_result:
            trace_event("query_cache_hit", {"request_id": request_id})
            logger.info("Cache hit for query", extra={"request_id": request_id})
            return QueryResponse(**cached_result)
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")

    # 2. Pipeline execution
    try:
        result = await agent.run(
            query=request.query,top_k=request.top_k or 5,use_reflection=request.use_reflection,
        )

        docs = [
            DocumentResponse(
                id=doc.get("id", f"doc_{i}"),
                score=float(doc.get("score", 1.0)),
                content=doc.get("content", ""),
            )
            for i, doc in enumerate(result.get("documents", []))
        ]

        response_data = QueryResponse(query=request.query,answer=result.get("answer", ""),documents=docs,
            reflection=result.get("reflection"),
            metadata={
                "request_id": request_id,
                **(result.get("metadata") or {}),
            },
        )

        trace_event("query_completed", {"request_id": request_id, "doc_count": len(docs)})

        # 3. Cache store
        try:
            await cache.set(cache_key, response_data.model_dump(), expire=3600)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

        return response_data

    except Exception as err:
        logger.error(f"Query execution failed: {err}", exc_info=True)
        trace_event("query_error", {"request_id": request_id, "error": str(err)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(err)}",
        )