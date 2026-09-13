import re
from typing import List, Optional, Tuple

from ...generation.LLMGenerator import LLMGenerator, get_llm_generator
from ...generation.prompt_temp import PromptTemplates
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class TaskPlannerWorkflow:
    # Task planner workflow for deconstructing tasks into ordered steps and matching them to available tools.


    def __init__(self, llm_generator: Optional[LLMGenerator] = None):
        self.llm = llm_generator or get_llm_generator()

    async def create_plan(self,task: str,available_tools: List[str],) -> List[str]:
        # Create an execution plan for the given task using the LLM.
        # The method returns a list of ordered steps that can be executed to complete the task.
        # Each step is a string describing an action to be taken, and may reference available tools.

        if not task or not task.strip():
            logger.warning("Empty task provided for plan generation")
            return []

        logger.info(f"Creating execution plan for task: '{task}'")

        prompt = PromptTemplates.agent_planning_prompt(task=task.strip(),available_tools=available_tools,)

        try:
            response = await self.llm.generate(prompt=prompt,temperature=0.3,max_tokens=500,)

            steps: List[str] = []
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Parse lines starting with numbers (1., 1-), asterisks, dashes, or step keywords
                if (
                    line[0].isdigit()
                    or line.startswith(("-", "*"))
                    or line.lower().startswith("step")
                ):
                    cleaned = re.sub(r"^(?:step\s*\d+[:.]?|\d+[.)\-*]|\*|-)\s*", "", line, flags=re.IGNORECASE).strip()
                    if cleaned:
                        steps.append(cleaned)

            # Fallback if the LLM output didn't use bullet or numerical delimiters
            if not steps and response.strip():
                steps = [line.strip() for line in response.split("\n") if line.strip()]

            logger.info(f"Generated task plan with {len(steps)} steps")
            return steps

        except Exception as e:
            logger.error(f"Task plan generation failed: {e}")
            raise AgentError(f"Task planner failed for task: '{task}'", original_error=e) from e

    def should_use_tool(self,step: str,available_tools: List[str],) -> Tuple[bool, str]:
        # Determine whether a specific step in the plan should be executed using an available tool.
        # The method returns a tuple (use_tool: bool, tool_name: str).
        # If use_tool is True, tool_name will contain the name of the tool to be used; otherwise, it will be an empty string.
        if not step or not step.strip():
            logger.warning("Empty step provided for tool usage check")
            return False, ""
        
        step_lower = step.lower()

        # 1. Knowledge base / internal retrieval checks
        kb_patterns = [r"\bretrieve\b", r"\binternal\b", r"\bdocuments?\b", r"\bknowledge\b", r"\bchunks?\b", r"\bfind in docs\b"]
        if any(re.search(pattern, step_lower) for pattern in kb_patterns):
            for name in ["knowledge_base", "retriever", "knowledge_base_retriever"]:
                if name in available_tools:
                    return True, name

        # 2. Mathematical / computational checks
        math_ops = [r"\+", r"\-", r"\*", r"/", r"//", r"%", r"\*\*"]
        math_words = [r"\bcalculate\b", r"\bcompute\b", r"\bsum\b", r"\bmultiply\b", r"\bdivide\b", r"\bpercentage\b", r"\barithmetic\b", r"\bformula\b"]
        if any(re.search(pattern, step_lower) for pattern in math_words) or any(re.search(pattern, step_lower) for pattern in math_ops):
            for name in ["math", "calculator", "math_calculator"]:
                if name in available_tools:
                    return True, name

        # 3. Live web search checks
        web_words = [r"\bweb\b", r"\binternet\b", r"\bgoogle\b", r"\blive\b", r"\bcurrent\b", r"\blastest\b", r"\bnews\b", r"\bonline\b", r"\bexternal search\b"]
        if any(re.search(pattern, step_lower) for pattern  in web_words):
            for name in ["external_search", "web_search"]:
                if name in available_tools:
                    return True, name

        # 4. Secondary fallback: broader search terms map to retrieval if available
        standalone_search = [r"\bsearch\b", r"\bfind\b"]
        if any(re.search(pattern, step_lower) for pattern in standalone_search):
                    for name in ["knowledge_base", "retriever", "external_search", "web_search"]:
                        if name in available_tools:
                            return True, name
        return False, ""