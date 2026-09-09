from typing import Any, Dict
import pytest

from src.agents.core_agent import AgentState, BaseAgent, Tool


class ConcreteTestTool(Tool):
    async def execute(self, *args, **kwargs) -> Any:
        return "tool_output"


class ConcreteTestAgent(BaseAgent):
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        state = self.create_state(query=query, **kwargs)
        state.add_thought("Processing query")
        state.answer = f"Processed: {query}"
        return state.to_dict()


def test_agent_state_initialization():
    state = AgentState(query="What is RAG?")
    assert state.query == "What is RAG?"
    assert state.iterations == 0
    assert state.should_continue is True
    assert state.thoughts == []
    assert state.tool_results == {}


def test_agent_state_thoughts_and_tools():
    state = AgentState(query="Compute metrics")
    state.add_thought("Need calculator tool")
    assert state.thoughts == ["Need calculator tool"]

    state.add_tool_result("calc", 42)
    assert state.tool_results["calc"] == 42


def test_agent_state_iteration_limit():
    state = AgentState(query="Test limit", max_iterations=2)
    assert state.should_continue is True

    state.increment_iteration()
    assert state.iterations == 1
    assert state.should_continue is True

    state.increment_iteration()
    assert state.iterations == 2
    assert state.should_continue is False


def test_agent_state_to_dict():
    state = AgentState(query="Sample")
    state_dict = state.to_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["query"] == "Sample"
    assert "iterations" in state_dict


@pytest.mark.asyncio
async def test_base_agent_create_state_and_run():
    agent = ConcreteTestAgent(name="test_agent")
    result = await agent.run("Hello world", context="Initial context")

    assert result["query"] == "Hello world"
    assert result["context"] == "Initial context"
    assert result["answer"] == "Processed: Hello world"
    assert result["thoughts"] == ["Processing query"]


@pytest.mark.asyncio
async def test_base_agent_should_continue():
    agent = ConcreteTestAgent(name="test_agent")
    state = agent.create_state(query="Loop check", max_iterations=1)

    assert await agent.should_continue_execution(state) is True

    state.increment_iteration()
    assert await agent.should_continue_execution(state) is False


@pytest.mark.asyncio
async def test_tool_abstraction():
    tool = ConcreteTestTool(name="dummy_tool", description="A dummy tool")
    assert tool.name == "dummy_tool"
    assert repr(tool) == "Tool(name=dummy_tool)"
    result = await tool.execute()
    assert result == "tool_output"