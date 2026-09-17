import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.query import router
from src.api.deps import get_cache_dependency, verify_api_key, get_rag_agent_dependency


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    agent.run.return_value = {
        "answer": "Machine learning is a subset of artificial intelligence.",
        "documents": [{"id": "doc1", "score": 0.95, "content": "Sample content"}],
        "reflection": {"score": 8.5},
        "metadata": {"iterations": 1},
    }
    return agent


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = True
    return cache

@pytest.fixture
def client(mock_agent, mock_cache):
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_cache_dependency] = lambda: mock_cache
    app.dependency_overrides[get_rag_agent_dependency] = lambda: mock_agent
    app.dependency_overrides[verify_api_key] = lambda: "test-key"

    return TestClient(app)


def test_query_validation_error(client):
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_query_success_mocked(client, mock_agent):
    response = client.post(
        "/query",
        json={"query": "What is machine learning?", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == mock_agent.run.return_value["answer"]
    assert len(data["documents"]) == 1
    assert data["documents"][0]["id"] == "doc1"
    assert "request_id" in data["metadata"]