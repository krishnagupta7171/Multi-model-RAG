import pytest

from src.agents.tools.math_tool import MathTool
from src.utils.exceptions import AgentError


@pytest.fixture
def math_tool():
    return MathTool()


@pytest.mark.asyncio
async def test_basic_arithmetic(math_tool):
    assert await math_tool.execute("2 + 2") == 4.0
    assert await math_tool.execute("10 - 4") == 6.0
    assert await math_tool.execute("3 * 5") == 15.0
    assert await math_tool.execute("15 / 3") == 5.0
    assert await math_tool.execute("7 // 2") == 3.0
    assert await math_tool.execute("7 % 3") == 1.0
    assert await math_tool.execute("2 ** 3") == 8.0


@pytest.mark.asyncio
async def test_precedence_and_grouping(math_tool):
    # Operator precedence
    assert await math_tool.execute("2 + 3 * 4") == 14.0
    # Explicit parentheses grouping
    assert await math_tool.execute("(2 + 3) * 4") == 20.0
    # Financial/percentage calculation
    result = await math_tool.execute("(5.1 - 4.2) / 4.2 * 100")
    assert pytest.approx(result, rel=1e-3) == 21.42857


@pytest.mark.asyncio
async def test_unary_operations(math_tool):
    assert await math_tool.execute("-5 + 10") == 5.0
    assert await math_tool.execute("+5 * -2") == -10.0


@pytest.mark.asyncio
async def test_division_by_zero(math_tool):
    with pytest.raises(AgentError, match="Division by zero"):
        await math_tool.execute("10 / 0")


@pytest.mark.asyncio
async def test_empty_or_invalid_inputs(math_tool):
    with pytest.raises(AgentError, match="non-empty string"):
        await math_tool.execute("")

    with pytest.raises(AgentError, match="non-empty string"):
        await math_tool.execute("   ")

    with pytest.raises(AgentError, match="non-empty string"):
        await math_tool.execute(None)  # type: ignore


@pytest.mark.asyncio
async def test_safety_rejects_arbitrary_code(math_tool):
    # Attempting to call functions or import modules must be rejected
    with pytest.raises(AgentError):
        await math_tool.execute("__import__('os').system('ls')")

    with pytest.raises(AgentError):
        await math_tool.execute("print('malicious')")

    with pytest.raises(AgentError):
        await math_tool.execute("x = 5")