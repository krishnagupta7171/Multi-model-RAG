from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from ..observability.logging import get_logger

logger = get_logger(__name__)


def _get_or_create(metric_type, name, doc, *args, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, doc, *args, **kwargs)


# Request metrics
http_requests_total = _get_or_create(
    Counter,
    "http_requests_total",
    "Total HTTP requests handled by the service",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = _get_or_create(
    Histogram,
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# RAG execution metrics
rag_queries_total = _get_or_create(
    Counter,
    "rag_queries_total",
    "Total RAG queries processed by the agent",
    ["status"],
)

rag_query_duration_seconds = _get_or_create(
    Histogram,
    "rag_query_duration_seconds",
    "Total RAG query execution duration in seconds",
)

rag_documents_retrieved = _get_or_create(
    Histogram,
    "rag_documents_retrieved",
    "Distribution of documents retrieved per query",
    buckets=[0, 1, 3, 5, 10, 20, 50],
)

rag_reflection_score = _get_or_create(
    Histogram,
    "rag_reflection_score",
    "Self-reflection quality critique score distribution",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

# Document ingestion metrics
ingestion_documents_total = _get_or_create(
    Counter,
    "ingestion_documents_total",
    "Total documents processed in the ingestion pipeline",
    ["status"],
)

ingestion_chunks_total = _get_or_create(
    Counter,
    "ingestion_chunks_total",
    "Total text and tabular chunks indexed",
)

ingestion_duration_seconds = _get_or_create(
    Histogram,
    "ingestion_duration_seconds",
    "Document ingestion processing latency in seconds",
)

# LLM provider metrics
llm_requests_total = _get_or_create(
    Counter,
    "llm_requests_total",
    "Total LLM generation calls",
    ["model", "status"],
)

llm_tokens_used = _get_or_create(
    Counter,
    "llm_tokens_used",
    "Cumulative token consumption by model",
    ["model", "type"],
)

llm_request_duration_seconds = _get_or_create(
    Histogram,
    "llm_request_duration_seconds",
    "Latency of LLM API completions in seconds",
    ["model"],
)

# Vector store metrics
vector_db_operations_total = _get_or_create(
    Counter,
    "vector_db_operations_total",
    "Total ChromaDB vector store operations",
    ["operation", "status"],
)

vector_db_operation_duration_seconds = _get_or_create(
    Histogram,
    "vector_db_operation_duration_seconds",
    "Latency of vector store retrieval and indexing operations",
    ["operation"],
)

# Cache layer metrics
cache_hits_total = _get_or_create(
    Counter,
    "cache_hits_total",
    "Total cache hit occurrences",
)

cache_misses_total = _get_or_create(
    Counter,
    "cache_misses_total",
    "Total cache miss occurrences",
)

# Concurrency & System gauges
active_requests = _get_or_create(
    Gauge,
    "active_requests",
    "Current active concurrent in-flight requests",
)

logger.info("Prometheus system metrics successfully initialized")