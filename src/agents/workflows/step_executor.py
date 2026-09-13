import re
from typing import Any, Dict, Optional

from ...agents.core_agent import AgentState
from ...generation.LLMGenerator import LLMGenerator, get_llm_generator
from ...observability.logging import get_logger

logger = get_logger(__name__)


class StepExecutorWorkflow:
    # Workflow for executing individual steps in a multi-step agent process.
    def __init__(self, llm_generator: Optional[LLMGenerator] = None):
        self.llm = llm_generator or get_llm_generator()

    async def execute_step(self,step: str,state: AgentState,tools: Dict[str, Any],) -> Dict[str, Any]:
        # Execute a single step in the workflow, determining
            #  if a tool is needed and invoking it or synthesizing the output with the LLM.
        # It returns a dictionary containing the success status, the tool used (if any), and the result of the step execution.
        # The method identifies the appropriate tool based on the step description, 
            # executes it if available, and handles any exceptions that may arise during execution. If no tool is matched, it synthesizes the step output using the LLM with the provided context and query from the state.
        # The method also logs the execution process for observability and debugging purposes.
    
        logger.info(f"Executing step: {step}")
        state.add_thought(f"Executing: {step}")

        tool_name = self._identify_tool(step, tools)

        if tool_name and tool_name in tools:
            tool = tools[tool_name]
            result = await self._execute_tool(tool, step, state)
            state.add_tool_result(tool_name, result)

            return {
                "success": True,
                "tool_used": tool_name,
                "result": result,
            }

        # No tool needed or matched; synthesize step output with LLM
        result = await self._process_with_llm(step, state)
        return {
            "success": True,
            "tool_used": None,
            "result": result,
        }

    def _identify_tool(self, step: str, tools: Dict[str, Any]) -> str:
        # Identify the appropriate tool to use for a given step based on keywords and patterns in the step description.
        # The method returns the name of the matched tool or an empty string if no tool is matched.
        # It uses regular expressions to match patterns for 
            # knowledge base retrieval, mathematical calculations, and live web searches, and checks against the available tools in the provided dictionary.
        # The method is designed to be extensible for additional tool matching logic as needed.
        
        step_lower = step.lower()

        # 1. Internal retrieval / Knowledge Base
        kb_patterns = [
            r"\bretrieve\b",
            r"\bkb\b",
            r"\binternal\b",
            r"\bdocuments?\b",
            r"\bknowledge\b",
            r"\bchunks?\b",
        ]
        if any(re.search(p, step_lower) for p in kb_patterns):
            for name in ["knowledge_base", "retriever", "knowledge_base_retriever"]:
                if name in tools:
                    return name

        # 2. Mathematical calculations
        math_patterns = [
            r"\bcalculate\b",
            r"\bcompute\b",
            r"\bsum\b",
            r"\bmultiply\b",
            r"\bdivide\b",
            r"\barithmetic\b",
            r"\bformula\b",
        ]
        math_ops = [r"\+", r"\-", r"\*", r"/", r"//", r"%", r"\*\*"]
        if any(re.search(p, step_lower) for p in math_patterns) or any(
            re.search(p, step_lower) for p in math_ops
        ):
            for name in ["math", "calculator", "math_calculator"]:
                if name in tools:
                    return name

        # 3. Live external web search
        web_patterns = [
            r"\bweb\b",
            r"\binternet\b",
            r"\bgoogle\b",
            r"\blive\b",
            r"\bcurrent\b",
            r"\blatest\b",
            r"\bnews\b",
            r"\bonline\b",
            r"\bexternal\b",
        ]
        if any(re.search(p, step_lower) for p in web_patterns):
            for name in ["external_search", "web_search"]:
                if name in tools:
                    return name

        # 4. Standalone search fallback
        if re.search(r"\bsearch\b", step_lower) or re.search(r"\bfind\b", step_lower):
            for name in ["knowledge_base", "retriever", "external_search", "web_search"]:
                if name in tools:
                    return name

        return ""

    async def _execute_tool(self,tool: Any,step: str,state: AgentState,) -> Any:
        # Execute the identified tool with the appropriate parameters based on the step description and the current state.
        # The method handles different types of tools, including mathematical calculators, knowledge base retrievers, and live web search tools.
        # It extracts the necessary input (e.g., mathematical expressions or queries) from the step description and invokes the tool's execute method.
        # The method also includes error handling to log and return any exceptions that occur during tool execution, ensuring that the workflow can continue gracefully even if a tool fails.
        # It returns the result of the tool execution, which may vary depending on the tool's functionality and the input provided.
        # The method is designed to be extensible for additional tool execution logic as needed, allowing for a flexible and robust step execution workflow.
        
        logger.debug(f"Executing tool: {tool.name}")
        tname = tool.name.lower()

        try:
            # Math / Calculator dispatch
            if any(k in tname for k in ["math", "calculator"]):
                expr = self._extract_math_expression(step)
                return await tool.execute(expression=expr)

            # Knowledge base / Vector search dispatch
            if any(k in tname for k in ["knowledge_base", "retriever"]):
                query = self._extract_query(step) or state.query
                return await tool.execute(query=query)

            # Live external web search dispatch
            if any(k in tname for k in ["external_search", "web_search"]):
                query = self._extract_query(step) or state.query
                return await tool.execute(query=query)

            return await tool.execute()

        except Exception as e:
            logger.error(f"Tool execution failed for '{tool.name}': {e}")
            return {"error": str(e)}

    def _extract_math_expression(self, step: str) -> str:
        # Extract a mathematical expression from the step description for calculator tools.
        # The method uses regular expressions to identify and extract valid mathematical expressions from the step text.
        # It returns the extracted expression as a string, or the original step if no valid expression is found.
        # The method is designed to be robust against various formats of mathematical expressions, including those with operators, parentheses, and numbers.
        # It is intended to be used in conjunction with the math tool execution logic to ensure that only valid expressions are passed to the calculator for evaluation.
        # The method can be extended or modified to handle additional formats or edge cases as needed for specific use cases.
        # The method also includes logging for debugging purposes, allowing developers to trace the extraction process and identify any issues with expression parsing.
        # It is important to note that the method does not validate the mathematical correctness of the expression; it only extracts a candidate expression based on the presence of digits and operators.
        # The actual evaluation and validation of the expression should be handled by the calculator tool itself.
        # The method is designed to be a utility function within the step execution workflow, providing a clear and concise way to extract mathematical expressions from natural language instructions.
        # The method can be used in various contexts where mathematical calculations are required, such as in multi-step workflows that involve data analysis, financial calculations, or scientific computations.
        # Overall, the method serves as a key component in enabling the agent to interpret and execute mathematical instructions effectively within the broader context of its reasoning and tool execution capabilities.
    
        match = re.search(r"[\d\.\s\+\-\*/\(\)\%]+", step)
        if match:
            candidate = match.group(0).strip()
            if any(char.isdigit() for char in candidate):
                return candidate
        return step

    def _extract_query(self, step: str) -> str:
        # Extract a query string from the step description for retrieval or search tools.
        # The method uses regular expressions to identify and extract potential query phrases from the step text.
        # It returns the extracted query as a string, or the original step if no valid query is found.
        # The method is designed to be flexible and can handle various formats of queries, including those
        # that may include keywords like "search", "find", or "retrieve".
        # It is intended to be used in conjunction with the retrieval or search tool execution logic to
        # ensure that only relevant queries are passed to the tools for execution.

        cleaned = re.sub(
            r"^(?:retrieve|search for|search|find|query)\s+",
            "",
            step,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned if cleaned else step

    async def _process_with_llm(self, step: str, state: AgentState) -> str:
        # Process a step using the LLM to synthesize an output based on the step description, the current state, and any retrieved context.
        # The method constructs a prompt that includes the step, the context from retrieved documents, and the original user query, and then invokes the LLM to generate a response.
        # It returns the synthesized output as a string, which can be used as the result of
        # the step execution when no specific tool is matched or required.

        docs = getattr(state, "retrieved_documents", []) or []
        context = "\n".join(
            [d.get("content", "") if isinstance(d, dict) else str(d) for d in docs]
        )

        prompt = (
            f"Complete the following step based on the available information.\n\n"
            f"Step: {step}\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {state.query}\n\n"
            f"Response:"
        )

        response = await self.llm.generate(prompt=prompt,temperature=0.3,max_tokens=400,)
        return response.strip()