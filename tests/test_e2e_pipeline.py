from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.api.main import app
from src.api.deps import (get_retriever_dependency,get_rag_agent_dependency,verify_api_key,)


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.add_texts.return_value = ["chunk_doc_0"]
    return retriever


@pytest.fixture
def mock_rag_agent():
    agent = AsyncMock()
    mock_doc = {
        "id": "chunk_doc_0",
        "score": 0.95,
        "content": "Multi-modal RAG uses combined vision and language contexts.",
    }
    agent.run.return_value = {
        "answer": "Multi-modal RAG enhances retrieval accuracy across text and visual data.",
        "documents": [mock_doc],
        "reflection": {"score": 9.2, "is_sufficient": True},
        "metadata": {"processing_time_sec": 0.42},
    }
    return agent


@pytest.fixture
def client(mock_retriever, mock_rag_agent):
    app.dependency_overrides[get_retriever_dependency] = lambda: mock_retriever
    app.dependency_overrides[get_rag_agent_dependency] = lambda: mock_rag_agent
    app.dependency_overrides[verify_api_key] = lambda: "test-auth-key"

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_full_pipeline_ingest_then_query(client, mock_retriever, mock_rag_agent):
    # Ingest document into the system
    mock_chunk = MagicMock()
    mock_chunk.text = "Multi-modal RAG uses combined vision and language contexts."
    mock_chunk.metadata = {"source": "manual_entry"}
    mock_chunk.chunk_id = "chunk_doc_0"

    with patch("src.api.routes.ingest.get_chunker") as mock_chunker_getter:
        mock_chunker = MagicMock()
        mock_chunker.chunk_documents.return_value = [mock_chunk]
        mock_chunker_getter.return_value = mock_chunker

        ingest_payload = {
            "text": "Multi-modal RAG uses combined vision and language contexts.",
            "metadata": {"source": "manual_entry"},
            "collection_name": "documents",
        }

        ingest_resp = client.post("/api/v1/ingest", json=ingest_payload)
        assert ingest_resp.status_code == 201
        ingest_data = ingest_resp.json()
        assert ingest_data["success"] is True
        assert ingest_data["num_documents"] == 1
        assert "chunk_doc_0" in ingest_data["document_ids"]
        mock_retriever.add_texts.assert_called_once()

    # Query the agent
    query_payload = {
        "query": "What does multi-modal RAG do?",
        "top_k": 3,
        "use_reflection": True,
        "stream": False,
    }

    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    query_data = query_resp.json()

    assert query_data["query"] == "What does multi-modal RAG do?"
    assert "Multi-modal RAG enhances retrieval" in query_data["answer"]
    assert len(query_data["documents"]) == 1
    assert query_data["documents"][0]["id"] == "chunk_doc_0"
    assert query_data["reflection"]["is_sufficient"] is True
    mock_rag_agent.run.assert_called_once()