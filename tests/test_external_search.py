from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.tools.external_search import ExternalSearchTool
from src.utils.exceptions import AgentError


@pytest.mark.asyncio
async def test_external_search_placeholder_mode():
    # Verify that the ExternalSearchTool returns placeholder 
    # results when no API key is provided.
    tool = ExternalSearchTool(api_key=None, max_results=3)
    results = await tool.execute(query="Python asyncio patterns")

    assert len(results) == 3
    assert "Python asyncio patterns" in results[0]["title"]
    assert results[0]["url"].startswith("https://example.com/result")
    assert "content" in results[0]
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.9


@pytest.mark.asyncio
async def test_external_search_with_mocked_client():
    # Verify that the ExternalSearchTool correctly uses a mocked Tavily client.
    tool = ExternalSearchTool(api_key="mock-api-key", max_results=2)

    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "FastAPI Production Guidelines",
                    "url": "https://fastapi.tiangolo.com",
                    "content": "Production patterns for async APIs...",
                    "score": 0.96,
                }
            ]
        }
    )
    tool._client = mock_client

    results = await tool.execute(query="FastAPI guidelines")

    assert len(results) == 1
    assert results[0]["title"] == "FastAPI Production Guidelines"
    assert results[0]["url"] == "https://fastapi.tiangolo.com"
    assert results[0]["score"] == 0.96
    mock_client.search.assert_awaited_once_with(
        query="FastAPI guidelines", max_results=2, search_depth="basic"
    )


@pytest.mark.asyncio
async def test_external_search_custom_max_results_override():
    """Verify that per-call max_results overrides default initialization."""
    tool = ExternalSearchTool(api_key=None, max_results=5)
    results = await tool.execute(query="multimodal RAG", max_results=2)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_external_search_empty_and_invalid_queries():
    # Verify that the ExternalSearchTool handles empty and non-string queries properly.
    tool = ExternalSearchTool(api_key=None)

    assert await tool.execute(query="") == []
    assert await tool.execute(query="   ") == []

    with pytest.raises(ValueError, match="Query must be a string"):
        await tool.execute(query=None)  # type: ignore


@pytest.mark.asyncio
async def test_external_search_api_failure_raises_agent_error():
    # Verify that the ExternalSearchTool raises AgentError on API failures.
    tool = ExternalSearchTool(api_key="mock-api-key")
    mock_client = MagicMock()
    mock_client.search = AsyncMock(side_effect=RuntimeError("Rate limit exceeded"))
    tool._client = mock_client

    with pytest.raises(AgentError, match="External search failed for query"):
        await tool.execute(query="failing search query")