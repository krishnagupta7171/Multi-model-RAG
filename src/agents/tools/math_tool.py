import ast
import operator
from typing import Any, Union

from ..core_agent import Tool
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class MathTool(Tool):
    # Tool for safely evaluating arithmetic expressions.
    # It supports +, -, *, /, //, %, **, and parentheses. 
    # It uses the ast module to parse expressions and only allows safe operations.

    # Explicit whitelist of safe arithmetic operations
    OPERATORS = {ast.Add: operator.add,ast.Sub: operator.sub,ast.Mult: operator.mul,ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,ast.Mod: operator.mod,ast.Pow: operator.pow,ast.USub: operator.neg,ast.UAdd: operator.pos,
    }

    def __init__(self,name: str = "math_calculator",
        description: str = "Perform safe arithmetic calculations (e.g., '(5.1 - 4.2) / 4.2 * 100', '1500 * 1.18'). Supports +, -, *, /, //, %, **, and parentheses.",
    ):
        
        super().__init__(name=name, description=description)

    async def execute(self, expression: str) -> float:
        # Safely evaluate a mathematical expression and return the result as a float.
        # It raises an AgentError for invalid expressions or unsupported operations.
        # If the expression is empty or not a string, it raises an AgentError.

        if not isinstance(expression, str) or not expression.strip():
            raise AgentError("Expression must be a non-empty string")

        logger.info(f"Evaluating math expression: {expression}")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._eval_node(tree.body)

            if not isinstance(result, (int, float)):
                raise ValueError(f"Evaluated expression did not return a number: {result}")

            logger.info(f"Math result: {result}")
            return float(result)

        except ZeroDivisionError as e:
            logger.error(f"Division by zero in expression '{expression}': {e}")
            raise AgentError(f"Math error: Division by zero in expression '{expression}'") from e
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {e}")
            raise AgentError(
                f"Invalid or unsupported mathematical expression: '{expression}'",
                original_error=e,
            ) from e

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        # Recursively evaluate an AST node representing a mathematical expression.
        # It only allows safe operations defined in the OPERATORS whitelist.
        #  Raises ValueError for unsupported nodes or operations.
        # if the node is a constant (number), return its value. and if it is a binary operation, recursively evaluate the left and right operands and apply the operator. 
        # If it is a unary operation, evaluate the operand and apply the operator. Otherwise, raise an error for unsupported syntax.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

            operand = self._eval_node(node.operand)
            return self.OPERATORS[op_type](operand)

        raise ValueError(f"Unsupported syntax or node type: {type(node).__name__}")