from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.workflows.task_planner import TaskPlannerWorkflow
from src.utils.exceptions import AgentError


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_create_plan_parses_formatted_steps(mock_llm):
    mock_llm.generate.return_value = (
        "1. Retrieve revenue figures from internal documents\n"
        "2. Calculate percentage growth using the math tool\n"
        "3. Search online for recent industry benchmarks\n"
        "4. Synthesize the final comparison report"
    )
    planner = TaskPlannerWorkflow(llm_generator=mock_llm)

    tools = ["knowledge_base", "math", "external_search"]
    steps = await planner.create_plan("Analyze revenue growth", tools)

    assert len(steps) == 4
    assert steps[0] == "Retrieve revenue figures from internal documents"
    assert steps[1] == "Calculate percentage growth using the math tool"
    assert steps[2] == "Search online for recent industry benchmarks"
    assert steps[3] == "Synthesize the final comparison report"


@pytest.mark.asyncio
async def test_create_plan_empty_task():
    planner = TaskPlannerWorkflow()
    assert await planner.create_plan("", ["math"]) == []
    assert await planner.create_plan("   ", ["math"]) == []


@pytest.mark.asyncio
async def test_create_plan_error_raises_agent_error(mock_llm):
    mock_llm.generate.side_effect = Exception("Groq connection timeout")
    planner = TaskPlannerWorkflow(llm_generator=mock_llm)

    with pytest.raises(AgentError, match="Task planner failed for task"):
        await planner.create_plan("Do something", ["math"])


def test_should_use_tool_knowledge_base():
    planner = TaskPlannerWorkflow()
    tools = ["knowledge_base", "math", "external_search"]

    use_tool, name = planner.should_use_tool("Retrieve document chunks for Q3 results", tools)
    assert use_tool is True
    assert name == "knowledge_base"


def test_should_use_tool_math():
    planner = TaskPlannerWorkflow()
    tools = ["knowledge_base", "math", "external_search"]

    use_tool, name = planner.should_use_tool("Calculate (120 - 80) / 80 * 100", tools)
    assert use_tool is True
    assert name == "math"


def test_should_use_tool_external_search():
    planner = TaskPlannerWorkflow()
    tools = ["knowledge_base", "math", "external_search"]

    use_tool, name = planner.should_use_tool("Search the live web for current stock prices", tools)
    assert use_tool is True
    assert name == "external_search"


def test_should_use_tool_no_match():
    planner = TaskPlannerWorkflow()
    tools = ["knowledge_base", "math"]

    use_tool, name = planner.should_use_tool("Summarize findings and write final answer", tools)
    assert use_tool is False
    assert name == ""