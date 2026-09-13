import re
from typing import Any, Dict, List, Optional

from ...generation.LLMGenerator import LLMGenerator, get_llm_generator
from ...generation.prompt_temp import PromptTemplates
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class SelfReflectionWorkflow:
    # Self-reflection workflow for evaluating generated answers.
    # The workflow uses an LLM to critique the answer, score its quality, and determine if it needs improvement.
    # The workflow can be integrated into an agent's generation process to enable iterative refinement of answers based on reflection feedback.

    def __init__(self,llm_generator: Optional[LLMGenerator] = None,min_reflection_score: float = 6.0,):
        # Initialize the self-reflection workflow with an LLM generator and a minimum reflection score threshold.

        self.llm = llm_generator or get_llm_generator()
        self.min_reflection_score = min_reflection_score

    async def reflect(self,query: str,answer: str,context: List[str],) -> Dict[str, Any]:
        # Perform self-reflection on the generated answer given the original query and retrieved context.
        # The method returns a dictionary containing the reflection score, critique text, and a boolean indicating whether the answer needs improvement.
        # The reflection score is a float between 0 and 10, where higher scores indicate better quality and factual grounding of the answer.
        # The critique text provides detailed feedback on the answer's strengths and weaknesses, and suggestions for improvement.
        # The method raises an AgentError if the reflection process fails due to LLM generation errors or other issues.

        logger.info("Executing self-reflection critique on generated answer")

        if not answer or not answer.strip():
            logger.warning("Empty answer provided for reflection")
            return {
                "score": 0.0,
                "critique": "Generated answer is empty.",
                "needs_improvement": True,
            }

        prompt = PromptTemplates.reflection_prompt(query=query,answer=answer,context=context,)

        try:
            response = await self.llm.generate(prompt=prompt,temperature=0.1,max_tokens=300,)

            score = self._extract_score(response)
            needs_improvement = score < self.min_reflection_score

            logger.info(f"Self-reflection finished: score={score}/10 (needs_improvement={needs_improvement})")

            return {"score": score,"critique": response.strip(),"needs_improvement": needs_improvement,}

        except Exception as e:
            logger.error(f"Self-reflection generation failed: {e}")
            raise AgentError(
                "Self-reflection workflow failed during evaluation",
                original_error=e,
            ) from e

    def _extract_score(self, critique_text: str) -> float:
        """Extract a numerical score from the critique text."""
        patterns = [
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"(?:score|rating|grade)\s*:\s*(\d+(?:\.\d+)?)",
            r"\b(\d+(?:\.\d+)?)\s*out\s*of\s*10\b",
        ]

        lowered = critique_text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)

        logger.warning("Score pattern not found in critique output; defaulting to 5.0")
        return 5.0

    def should_retry(self,reflection: Dict[str, Any],max_retries: int = 2,current_retry: int = 0,) -> bool:
        # Determine whether to retry generation based on reflection score and retry count.
        # If the reflection score is below the minimum threshold and the maximum retries have not been reached,
        #  it will return True to indicate that a retry is needed; otherwise, it will return False.
        
        if current_retry >= max_retries:
            logger.info(f"Max retries ({max_retries}) reached; stopping retry loop")
            return False

        score = reflection.get("score", 10.0)
        if score < self.min_reflection_score:
            logger.info(f"Reflection score {score} < {self.min_reflection_score}; triggering retry")
            return True

        return False