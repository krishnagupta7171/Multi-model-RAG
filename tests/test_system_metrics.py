from prometheus_client import REGISTRY

from src.observability.system_metrics import (active_requests,cache_hits_total,cache_misses_total,http_requests_total,llm_tokens_used,rag_documents_retrieved,rag_queries_total,)


def test_metric_registration_in_prometheus_registry():
    collector_names = set(REGISTRY._names_to_collectors.keys())
    assert "http_requests_total" in collector_names
    assert "rag_queries_total" in collector_names
    assert "active_requests" in collector_names
    assert "cache_hits_total" in collector_names


def test_counter_increments():
    init_hits = cache_hits_total._value.get()
    cache_hits_total.inc()
    assert cache_hits_total._value.get() == init_hits + 1

    init_misses = cache_misses_total._value.get()
    cache_misses_total.inc(2)
    assert cache_misses_total._value.get() == init_misses + 2


def test_labeled_counter_increments():
    rag_queries_total.labels(status="success").inc()
    rag_queries_total.labels(status="error").inc()

    http_requests_total.labels(method="POST", endpoint="/query", status="200").inc()
    llm_tokens_used.labels(model="gemini", type="prompt").inc(150)


def test_gauge_operations():
    active_requests.set(5)
    assert active_requests._value.get() == 5
    active_requests.inc()
    assert active_requests._value.get() == 6
    active_requests.dec(2)
    assert active_requests._value.get() == 4


def test_histogram_observations():
    rag_documents_retrieved.observe(4)
    rag_documents_retrieved.observe(8)