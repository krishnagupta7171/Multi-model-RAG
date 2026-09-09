"""Core agent base class, execution state, and tool abstractions."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    # Execution state container for the agent's reasoning and actions.
    # This class holds the current state of the agent's execution, including the query, context,
    #  plan, thoughts, retrieved documents, tool results, and output. It also tracks iterations and flow control metadata.
    # Input
    query: str = ""
    context: str = ""

    # Planning and reasoning
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    thoughts: List[str] = field(default_factory=list)

    # Retrieved context and tool execution
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)

    # Output
    answer: str = ""
    reflection: str = ""
    confidence: float = 0.0

    # Flow control metadata
    iterations: int = 0
    max_iterations: int = 10
    should_continue: bool = True

    def add_thought(self, thought: str) -> None:
        # Add a new thought to the agent's reasoning process.
        # Log the thought for debugging and traceability.
#At every step of the loop, the agent uses that thinking trace to evaluate three things:
#What do I have right now? (Current state, extracted numbers, chunks retrieved)
#What is still missing? (Gaps, unverified facts, further calculations)
#What function or tool should I run next? (Retrieve more docs, call the calculator, run code, or synthesize the final answer)
        self.thoughts.append(thought)
        logger.debug(f"Agent thought: {thought}")

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        # Store the result of a tool execution in the agent's state.
        self.tool_results[tool_name] = result
        logger.debug(f"Tool result stored: {tool_name}")

    def increment_iteration(self) -> None:
        # Increment the iteration counter and check if the maximum iterations have been reached.
        self.iterations += 1
        if self.iterations >= self.max_iterations:
            self.should_continue = False
            logger.warning(f"Max iterations ({self.max_iterations}) reached")

    def to_dict(self) -> Dict[str, Any]:
        # Convert state container to dictionary representation.
        return asdict(self)


class Tool(ABC):
    # Abstract base class for tools that the agent
    # can use to perform specific actions or retrieve information.

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        # Execute the tool's underlying operation. 
        pass

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"


class BaseAgent(ABC):
    # Abstract base class for agents that
    #  orchestrate reasoning, retrieval, and tool execution.
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        # Execute the complete agent workflow.
        pass

    def create_state(self, query: str, **kwargs) -> AgentState:
        # Initialize and populate execution state for a query.
        state = AgentState(query=query)
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    async def should_continue_execution(self, state: AgentState) -> bool:
        # Evaluate whether execution should proceed to the next iteration.
        if not state.should_continue or state.iterations >= state.max_iterations:
            return False
        return True