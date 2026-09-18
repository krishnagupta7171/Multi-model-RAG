from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.deps import get_retriever_dependency, verify_api_key
from src.api.routes.ingest import router


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.add_texts.return_value = ["chunk_1"]
    return retriever


@pytest.fixture
def client(mock_retriever):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_retriever_dependency] = lambda: mock_retriever
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    return TestClient(app)


def test_ingest_missing_input(client):
    response = client.post("/ingest", json={})
    assert response.status_code == 400
    assert "Either file_path or text must be provided" in response.json()["detail"]


def test_ingest_text_success(client, mock_retriever):
    mock_chunk = MagicMock()
    mock_chunk.text = "Sample chunk text"
    mock_chunk.metadata = {"source": "unit_test"}
    mock_chunk.chunk_id = "chunk_1"

    with patch("src.api.routes.ingest.get_chunker") as mock_get_chunker:
        chunker_instance = MagicMock()
        chunker_instance.chunk_documents.return_value = [mock_chunk]
        mock_get_chunker.return_value = chunker_instance

        payload = {
            "text": "Artificial intelligence and machine learning pipelines.",
            "metadata": {"source": "unit_test"},
        }
        response = client.post("/ingest", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["num_documents"] == 1
        assert data["num_chunks"] == 1
        assert "chunk_1" in data["document_ids"]
        mock_retriever.add_texts.assert_called_once()