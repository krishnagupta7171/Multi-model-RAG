import ast
import builtins
from io import StringIO
import sys
from typing import Any, Dict

from ...agents.core_agent import Tool
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class CodeExecutorTool(Tool):
    # Explicit whitelist of safe built-in functions
    ALLOWED_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "print",
        "map",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }

    def __init__(self):
        super().__init__(
            name="code_executor",
            description="Execute sandboxed Python code for arithmetic, list transforms, and analytical calculations.",
        )

    async def execute(self, code: str, **kwargs: Any) -> Dict[str, Any]:

    
        if not code or not code.strip():
            return {
                "success": True,
                "output": "",
                "variables": "{}",
                "error": None,
            }

        logger.info("Executing sandboxed Python snippet")

        try:
            #  Syntax analysis validation
            ast.parse(code)

            #  Setup safe builtins dictionary
            safe_builtins = {}
            for name in self.ALLOWED_BUILTINS:
                if hasattr(builtins, name):
                    safe_builtins[name] = getattr(builtins, name)

            namespace: Dict[str, Any] = {"__builtins__": safe_builtins}

            # Capture stdout during execution
            stdout = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout

            try:
                exec(code, namespace)
                output = stdout.getvalue()

                # Extract computed variables ignore private and builtins
                variables = {
                    k: v
                    for k, v in namespace.items()
                    if not k.startswith("_") and k != "__builtins__"
                }

                logger.info("Python snippet executed successfully")
                return {
                    "success": True,
                    "output": output.strip(),
                    "variables": str(variables),
                    "error": None,
                }
            finally:
                sys.stdout = old_stdout

        except SyntaxError as e:
            logger.error(f"Syntax error in code execution: {e}")
            raise AgentError(f"Syntax error in code: {e}", original_error=e) from e

        except Exception as e:
            logger.error(f"Code execution runtime error: {e}")
            raise AgentError(f"Code execution failed: {e}", original_error=e) from e