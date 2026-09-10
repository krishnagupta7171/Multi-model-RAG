from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.tools.knowledge_base_tool import KnowledgeBaseTool


@pytest.mark.asyncio
async def test_knowledge_base_tool_vector_retriever():
    mock_retriever = MagicMock()
    mock_retriever.similarity_search = AsyncMock(return_value=[
        ("doc_1", 0.95, "Vector search result content"),
    ])

    tool = KnowledgeBaseTool(retriever=mock_retriever, top_k=3)
    results = await tool.execute(query="test query")

    assert len(results) == 1
    assert results[0]["id"] == "doc_1"
    assert results[0]["score"] == 0.95
    assert results[0]["content"] == "Vector search result content"
    mock_retriever.similarity_search.assert_awaited_once_with(query="test query", top_k=3)


@pytest.mark.asyncio
async def test_knowledge_base_tool_hybrid_retriever():
    mock_hybrid = MagicMock(spec=["search"])
    mock_hybrid.search = AsyncMock(return_value=[
        ("doc_hybrid", 0.88, "Hybrid search result content"),
    ])

    tool = KnowledgeBaseTool(retriever=mock_hybrid, top_k=5)
    results = await tool.execute(query="hybrid query", top_k=2)

    assert len(results) == 1
    assert results[0]["id"] == "doc_hybrid"
    mock_hybrid.search.assert_awaited_once_with(query="hybrid query", top_k=2)


@pytest.mark.asyncio
async def test_knowledge_base_tool_empty_query():
    mock_retriever = MagicMock()
    tool = KnowledgeBaseTool(retriever=mock_retriever)

    assert await tool.execute(query="") == []
    assert await tool.execute(query="   ") == []


@pytest.mark.asyncio
async def test_knowledge_base_tool_non_string_query():
    mock_retriever = MagicMock()
    tool = KnowledgeBaseTool(retriever=mock_retriever)

    with pytest.raises(ValueError, match="Query must be a string"):
        await tool.execute(query=None)  # type: ignore


@pytest.mark.asyncio
async def test_knowledge_base_tool_invalid_retriever():
    tool = KnowledgeBaseTool(retriever=object())

    with pytest.raises(AttributeError, match="must implement 'similarity_search' or 'search'"):
        await tool.execute(query="valid query")