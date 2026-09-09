from unittest.mock import AsyncMock, MagicMock
import pytest

from src.retrieval.hybrid_retriever import (BM25Retriever,HybridRetriever,MultiQueryRetriever,)
from src.utils.exceptions import RetrievalError

#BM25Retriever Tests
def test_bm25_search_without_indexing_raises():
    retriever = BM25Retriever()
    with pytest.raises(RetrievalError):
        retriever.search("test query")


def test_bm25_empty_query():
    retriever = BM25Retriever()
    retriever.index_documents([("d1", "sample content")])
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_bm25_index_and_search():
    retriever = BM25Retriever()
    docs = [
        ("doc1", "FastAPI is a modern web framework for Python"),
        ("doc2", "Docker containers package software into standard units"),
        ("doc3", "Python is an interpreted high-level general-purpose language"),
    ]
    retriever.index_documents(docs)

    results = retriever.search("FastAPI Python", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "doc1"


# HybridRetriever Tests
@pytest.mark.asyncio
async def test_hybrid_search():
    mock_vector = MagicMock()
    mock_vector.similarity_search = AsyncMock(return_value=[
        ("doc1", 0.9, "FastAPI web framework"),
        ("doc2", 0.7, "Docker containers"),
    ])
    mock_vector.add_texts = AsyncMock()

    hybrid = HybridRetriever(vector_retriever=mock_vector, alpha=0.5)

    docs = [
        ("doc1", "FastAPI web framework"),
        ("doc2", "Docker containers"),
    ]
    hybrid.bm25_retriever.index_documents(docs)

    results = await hybrid.search(query="FastAPI", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "doc1"
    assert "FastAPI" in results[0][2]


#MultiQueryRetriever Tests 

@pytest.mark.asyncio
async def test_multi_query_retriever_no_llm():
    mock_vector = MagicMock()
    mock_vector.similarity_search = AsyncMock(return_value=[
        ("doc1", 0.95, "LangGraph workflows"),
    ])

    retriever = MultiQueryRetriever(base_retriever=mock_vector, llm_client=None)
    results = await retriever.search("workflows", top_k=1)

    assert len(results) == 1
    assert results[0][0] == "doc1"


@pytest.mark.asyncio
async def test_multi_query_retriever_with_llm():
    mock_vector = MagicMock()
    mock_vector.similarity_search = AsyncMock(side_effect=[
        [("doc1", 0.8, "Query 1 doc")],
        [("doc2", 0.9, "Query 2 doc")],
    ])

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="variation query one\nvariation query two")

    retriever = MultiQueryRetriever(base_retriever=mock_vector, llm_client=mock_llm)
    results = await retriever.search("original query", num_queries=2, top_k=5)

    assert len(results) == 2
    assert results[0][0] == "doc2"
    assert results[1][0] == "doc1"