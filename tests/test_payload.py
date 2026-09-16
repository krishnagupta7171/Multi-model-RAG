import pytest
from pydantic import ValidationError

from src.api.payload import (BatchIngestRequest,DocumentResponse,HealthResponse,IngestRequest,IngestResponse,QueryRequest,QueryResponse,)


def test_query_request_valid():
    req = QueryRequest(query="Explain contrastive learning", top_k=4)
    assert req.query == "Explain contrastive learning"
    assert req.top_k == 4
    assert req.use_reflection is True
    assert req.stream is False


def test_query_request_validation_failure():
    # Empty query should fail
    with pytest.raises(ValidationError):
        QueryRequest(query="")

    # top_k < 1 should fail
    with pytest.raises(ValidationError):
        QueryRequest(query="valid query", top_k=0)


def test_ingest_request_defaults():
    req = IngestRequest(text="Test chunk content")
    assert req.collection_name == "documents"
    assert req.metadata == {}
    assert req.file_path is None


def test_query_response_serialization():
    doc = DocumentResponse(id="chunk_1", score=0.88, content="Chunk text")
    resp = QueryResponse(query="test query",answer="test answer",documents=[doc],metadata={"tokens": 120},)
    data = resp.model_dump()
    assert data["answer"] == "test answer"
    assert len(data["documents"]) == 1
    assert data["documents"][0]["score"] == 0.88


def test_health_response():
    health = HealthResponse(status="healthy",version="1.0.0",dependencies={"redis": "healthy", "vector_db": "healthy"},)
    assert health.status == "healthy"
    assert health.dependencies["redis"] == "healthy"