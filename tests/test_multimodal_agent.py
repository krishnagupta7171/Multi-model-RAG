from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.multimodal_agent import MultimodalRAGAgent
from src.utils.exceptions import AgentError


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.similarity_search = AsyncMock(
        return_value=[
            ("doc_1", 0.95, "LangGraph orchestrates cyclical agent state graphs."),
            ("doc_2", 0.82, "FastAPI handles non-blocking asynchronous calls."),
        ]
    )
    return retriever


@pytest.fixture
def mock_reranker():
    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=[
            ("doc_1", 0.98, "LangGraph orchestrates cyclical agent state graphs."),
        ]
    )
    return reranker


@pytest.mark.asyncio
async def test_multimodal_agent_run_success(mock_retriever, mock_reranker):
    agent = MultimodalRAGAgent(
        retriever=mock_retriever,
        reranker=mock_reranker,
        use_reflection=True,
    )

    result = await agent.run(query="What is LangGraph?")

    assert result["query"] == "What is LangGraph?"
    assert "Synthesized response" in result["answer"]
    assert len(result["documents"]) == 1
    assert result["documents"][0]["id"] == "doc_1"
    assert result["metadata"]["iterations"] == 1
    assert result["reflection"]["score"] >= 6.0


@pytest.mark.asyncio
async def test_multimodal_agent_without_reflection(mock_retriever):
    agent = MultimodalRAGAgent(
        retriever=mock_retriever,
        reranker=None,
        use_reflection=False,
    )

    result = await agent.run(query="Explain vectors")

    assert len(result["documents"]) == 2
    assert result["reflection"] == {}
    assert result["metadata"]["iterations"] == 0


@pytest.mark.asyncio
async def test_multimodal_agent_reflection_retry():
    mock_retriever = MagicMock()
    mock_retriever.similarity_search = AsyncMock(return_value=[("d1", 0.9, "Context text")])

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=["Draft 1 Answer", "Improved Final Answer"])
    # First reflection triggers retry (score 4.0 < 6.0), second reflection succeeds (9.0)
    mock_llm.reflect = AsyncMock(
        side_effect=[
            {"score": 4.0, "critique": "Needs more detail."},
            {"score": 9.0, "critique": "Grounded and complete."},
        ]
    )

    agent = MultimodalRAGAgent(
        retriever=mock_retriever,
        llm_client=mock_llm,
        use_reflection=True,
        min_reflection_score=6.0,
        max_retries=2,
    )

    result = await agent.run(query="Test reflection loop")

    assert result["answer"] == "Improved Final Answer"
    assert result["metadata"]["iterations"] == 2
    assert result["reflection"]["score"] == 9.0


@pytest.mark.asyncio
async def test_multimodal_agent_empty_query():
    agent = MultimodalRAGAgent(retriever=MagicMock())
    with pytest.raises(AgentError, match="Agent query must be a non-empty string"):
        await agent.run(query="   ")