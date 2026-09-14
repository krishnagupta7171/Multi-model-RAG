import pytest

from src.agents.tools.code_executor import CodeExecutorTool
from src.utils.exceptions import AgentError


@pytest.mark.asyncio
async def test_code_executor_basic_arithmetic():
    tool = CodeExecutorTool()
    code = """
x = 15
y = 30
total = x + y
print(f"Total: {total}")
"""
    result = await tool.execute(code)
    assert result["success"] is True
    assert result["output"] == "Total: 45"
    assert "'total': 45" in result["variables"]


@pytest.mark.asyncio
async def test_code_executor_allowed_builtins():
    tool = CodeExecutorTool()
    code = """
nums = [4, 1, 9, 2]
highest = max(nums)
total = sum(nums)
"""
    result = await tool.execute(code)
    assert result["success"] is True
    assert "'highest': 9" in result["variables"]
    assert "'total': 16" in result["variables"]


@pytest.mark.asyncio
async def test_code_executor_restricted_imports():
    tool = CodeExecutorTool()
    # __import__ is not in ALLOWED_BUILTINS
    code = "import os; os.listdir('.')"
    with pytest.raises(AgentError):
        await tool.execute(code)


@pytest.mark.asyncio
async def test_code_executor_syntax_error():
    tool = CodeExecutorTool()
    bad_code = "for x in [1, 2"
    with pytest.raises(AgentError, match="Syntax error"):
        await tool.execute(bad_code)


@pytest.mark.asyncio
async def test_code_executor_empty_input():
    tool = CodeExecutorTool()
    result = await tool.execute("")
    assert result["success"] is True
    assert result["output"] == ""