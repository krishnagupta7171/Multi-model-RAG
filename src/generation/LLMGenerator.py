from typing import Any, AsyncIterator, Dict, List, Optional
from groq import AsyncGroq

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import GenerationError

logger = get_logger(__name__)


class LLMGenerator:
    #LLMGenerator is responsible for interacting with the Groq API to generate text completions, chat responses, and stream outputs based on prompts and context.

    def __init__(self,api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,max_tokens: Optional[int] = None,):

        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.llm_model
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature)
        self.max_tokens = max_tokens or settings.llm_max_tokens

        self.client = AsyncGroq(api_key=self.api_key)

    async def generate(self,prompt: str,system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,**kwargs,) -> str:
        #Generate text completion.
        try:
            logger.debug(f"Generating completion with model {self.model}")

            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stop=stop_sequences or None,
                **kwargs,
            )

            generated_text = response.choices[0].message.content or ""
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise GenerationError(
                "LLM generation failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )

    async def generate_with_context(self,query: str,context: List[str],system: Optional[str] = None,**kwargs,) -> str:
        #Generate a response based on the provided context and query.
        context_text = "\n\n".join(
            [f"Context {i + 1}:\n{chunk}" for i, chunk in enumerate(context)]
        )

        prompt = f"""Based on the following context, please answer the question.\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"""

        return await self.generate(prompt=prompt, system=system, **kwargs)

    async def stream(self,prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,**kwargs,) -> AsyncIterator[str]:
        #Stream generated tokens in real time.
        try:
            logger.debug(f"Streaming completion with model {self.model}")

            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response_stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise GenerationError(
                "LLM streaming failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )

    async def chat(self,messages: List[Dict[str, str]],system: Optional[str] = None,**kwargs,) -> str:
        #Chat with the model using a list of messages, optionally including a system message.
        try:
            full_messages: List[Dict[str, str]] = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Groq chat failed: {e}")
            raise GenerationError(
                "Chat completion failed",
                details={"model": self.model, "error": str(e)},
                original_error=e,
            )


_generator: Optional[LLMGenerator] = None


def get_llm_generator() -> LLMGenerator:
    #Get a singleton instance of LLMGenerator.
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator