from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.core_agent import AgentState
from src.agents.workflows.step_executor import StepExecutorWorkflow


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Synthesized step completion.")
    return llm


@pytest.fixture
def mock_tools():
    kb_tool = MagicMock()
    kb_tool.name = "knowledge_base"
    kb_tool.execute = AsyncMock(return_value=[{"id": "doc1", "content": "RAG chunk"}])

    math_tool = MagicMock()
    math_tool.name = "math"
    math_tool.execute = AsyncMock(return_value=42.0)

    web_tool = MagicMock()
    web_tool.name = "external_search"
    web_tool.execute = AsyncMock(return_value=[{"title": "Latest news"}])

    return {
        "knowledge_base": kb_tool,
        "math": math_tool,
        "external_search": web_tool,
    }


@pytest.mark.asyncio
async def test_execute_step_with_knowledge_base(mock_tools):
    executor = StepExecutorWorkflow()
    state = AgentState(query="What is RAG?")

    result = await executor.execute_step(step="Retrieve internal documents about RAG architecture",state=state,tools=mock_tools,)

    assert result["success"] is True
    assert result["tool_used"] == "knowledge_base"
    assert result["result"] == [{"id": "doc1", "content": "RAG chunk"}]
    assert state.tool_results["knowledge_base"] == [{"id": "doc1", "content": "RAG chunk"}]
    assert any("Executing:" in t for t in state.thoughts)


@pytest.mark.asyncio
async def test_execute_step_with_math_tool(mock_tools):
    executor = StepExecutorWorkflow()
    state = AgentState(query="Calculate budget")

    result = await executor.execute_step(step="Compute 10 * 4 + 2",state=state,tools=mock_tools,)

    assert result["success"] is True
    assert result["tool_used"] == "math"
    assert result["result"] == 42.0
    mock_tools["math"].execute.assert_awaited_once_with(expression="10 * 4 + 2")


@pytest.mark.asyncio
async def test_execute_step_with_web_search(mock_tools):
    executor = StepExecutorWorkflow()
    state = AgentState(query="Latest stock trends")

    result = await executor.execute_step(step="Search the live internet for stock movements",state=state,tools=mock_tools,)

    assert result["success"] is True
    assert result["tool_used"] == "external_search"
    assert result["result"] == [{"title": "Latest news"}]


@pytest.mark.asyncio
async def test_execute_step_fallback_to_llm(mock_llm, mock_tools):
    executor = StepExecutorWorkflow(llm_generator=mock_llm)
    state = AgentState(query="What should we name the project?")

    result = await executor.execute_step(step="Summarize findings and write final executive recommendation",state=state,tools=mock_tools,)

    assert result["success"] is True
    assert result["tool_used"] is None
    assert result["result"] == "Synthesized step completion."
    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_failure_returns_graceful_error(mock_tools):
    mock_tools["math"].execute.side_effect = ZeroDivisionError("division by zero")
    executor = StepExecutorWorkflow()
    state = AgentState(query="Faulty math")

    result = await executor.execute_step(step="Compute 10 / 0",state=state,tools=mock_tools,)

    assert result["success"] is True
    assert result["tool_used"] == "math"
    assert "error" in result["result"]
    assert "division by zero" in result["result"]["error"]