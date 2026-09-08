from unittest.mock import AsyncMock, MagicMock
import pytest

from src.retrieval.query_transform import (QueryTransformer,HyDETransformer,QueryDecomposer,get_query_transformer,)
from src.utils.exceptions import RetrievalError


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.generate = AsyncMock()
    return client



#query transformer tests
@pytest.mark.asyncio
async def test_rewrite_query_without_llm():
    transformer = QueryTransformer(llm_client=None)
    result = await transformer.rewrite_query("what is rag")
    assert result == "what is rag"


@pytest.mark.asyncio
async def test_rewrite_query_success(mock_llm_client):
    mock_llm_client.generate.return_value = "explain retrieval augmented generation\nsecond variation"
    transformer = QueryTransformer(llm_client=mock_llm_client)

    result = await transformer.rewrite_query("rag info")
    assert result == "explain retrieval augmented generation"
    mock_llm_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_error(mock_llm_client):
    mock_llm_client.generate.side_effect = Exception("LLM call timeout")
    transformer = QueryTransformer(llm_client=mock_llm_client)

    result = await transformer.rewrite_query("rag info")
    assert result == "rag info"


@pytest.mark.asyncio
async def test_expand_query_success(mock_llm_client):
    mock_llm_client.generate.return_value = "query variation 1\nquery variation 2"
    transformer = QueryTransformer(llm_client=mock_llm_client)

    results = await transformer.expand_query("machine learning")
    assert results == ["machine learning", "query variation 1", "query variation 2"]


def test_add_context():
    transformer = QueryTransformer()
    assert transformer.add_context("What is it?", "") == "What is it?"
    result = transformer.add_context("What is it?", "We are talking about Python.")
    assert "Context: We are talking about Python." in result
    assert "Question: What is it?" in result


def test_extract_keywords():
    transformer = QueryTransformer()
    keywords = transformer.extract_keywords("What is the speed of an unladen swallow?")
    assert "speed" in keywords
    assert "unladen" in keywords
    assert "swallow" in keywords
    assert "what" not in keywords
    assert "the" not in keywords



#HyDETransformer Tests
@pytest.mark.asyncio
async def test_hyde_generate_success(mock_llm_client):
    mock_llm_client.generate.return_value = "Retrieval Augmented Generation combines vector search with LLMs."
    hyde = HyDETransformer(llm_client=mock_llm_client)

    doc = await hyde.generate_hypothetical_document("How does RAG work?")
    assert "Retrieval Augmented Generation" in doc


@pytest.mark.asyncio
async def test_hyde_generate_failure_raises(mock_llm_client):
    mock_llm_client.generate.side_effect = Exception("Service unavailable")
    hyde = HyDETransformer(llm_client=mock_llm_client)

    with pytest.raises(RetrievalError):
        await hyde.generate_hypothetical_document("Any query")


#query decomposer tests
@pytest.mark.asyncio
async def test_decompose_success(mock_llm_client):
    mock_llm_client.generate.return_value = "What is Docker?\nWhat is Kubernetes?\nHow do they differ?"
    decomposer = QueryDecomposer(llm_client=mock_llm_client)

    sub_queries = await decomposer.decompose("Compare Docker and Kubernetes architecture")
    assert len(sub_queries) == 3
    assert sub_queries[0] == "What is Docker?"


def test_get_query_transformer_singleton():
    t1 = get_query_transformer()
    t2 = get_query_transformer()
    assert t1 is t2