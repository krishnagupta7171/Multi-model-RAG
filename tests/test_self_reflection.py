from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.workflows.self_reflection import SelfReflectionWorkflow
from src.utils.exceptions import AgentError


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_reflect_success_high_score(mock_llm):
    mock_llm.generate.return_value = "The answer is fully grounded.\nScore: 9/10"
    workflow = SelfReflectionWorkflow(llm_generator=mock_llm, min_reflection_score=6.0)

    result = await workflow.reflect(
        query="What is RAG?",
        answer="RAG combines retrieval with LLMs.",
        context=["RAG combines retrieval with LLMs."],
    )

    assert result["score"] == 9.0
    assert result["needs_improvement"] is False
    assert "Score: 9/10" in result["critique"]


@pytest.mark.asyncio
async def test_reflect_low_score_triggers_improvement(mock_llm):
    mock_llm.generate.return_value = "The answer hallucinates details.\nRating: 4.5"
    workflow = SelfReflectionWorkflow(llm_generator=mock_llm, min_reflection_score=6.0)

    result = await workflow.reflect(
        query="What is Chroma?",
        answer="Chroma is a web framework.",
        context=["Chroma is a vector database."],
    )

    assert result["score"] == 4.5
    assert result["needs_improvement"] is True


@pytest.mark.asyncio
async def test_reflect_empty_answer():
    workflow = SelfReflectionWorkflow(min_reflection_score=6.0)
    result = await workflow.reflect(query="Test", answer="", context=[])

    assert result["score"] == 0.0
    assert result["needs_improvement"] is True


@pytest.mark.asyncio
async def test_reflect_error_raises_agent_error(mock_llm):
    mock_llm.generate.side_effect = Exception("LLM call timeout")
    workflow = SelfReflectionWorkflow(llm_generator=mock_llm)

    with pytest.raises(AgentError, match="Self-reflection workflow failed"):
        await workflow.reflect(query="Test", answer="Answer", context=[])


def test_should_retry_logic():
    workflow = SelfReflectionWorkflow(min_reflection_score=6.0)

    assert workflow.should_retry({"score": 4.0}, max_retries=2, current_retry=1) is True

    assert workflow.should_retry({"score": 8.0}, max_retries=2, current_retry=1) is False

    
    assert workflow.should_retry({"score": 4.0}, max_retries=2, current_retry=2) is False