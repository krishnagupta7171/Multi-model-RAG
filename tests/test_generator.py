from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.generation.LLMGenerator import LLMGenerator
from src.utils.exceptions import GenerationError


@pytest.fixture
def mock_groq_client():
    with patch("src.generation.LLMGenerator.AsyncGroq") as mock_groq:
        client_instance = MagicMock()
        mock_groq.return_value = client_instance
        yield client_instance


@pytest.mark.asyncio
async def test_generator_generate_success(mock_groq_client):
    mock_choice = MagicMock()
    mock_choice.message.content = "Generated answer"
    mock_response = MagicMock(choices=[mock_choice])

    mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

    generator = LLMGenerator(api_key="mock_key", model="llama-3.3-70b-versatile")
    result = await generator.generate(prompt="What is RAG?", system="You are an expert.")

    assert result == "Generated answer"
    mock_groq_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_generator_generate_with_context(mock_groq_client):
    mock_choice = MagicMock()
    mock_choice.message.content = "RAG combines retrieval with LLMs."
    mock_response = MagicMock(choices=[mock_choice])

    mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

    generator = LLMGenerator(api_key="mock_key")
    result = await generator.generate_with_context(
        query="Explain RAG",
        context=["RAG fetches chunks.", "RAG synthesizes with an LLM."],
    )

    assert result == "RAG combines retrieval with LLMs."
    call_args = mock_groq_client.chat.completions.create.call_args[1]
    prompt_sent = call_args["messages"][-1]["content"]
    assert "Context 1:\nRAG fetches chunks." in prompt_sent
    assert "Question: Explain RAG" in prompt_sent


@pytest.mark.asyncio
async def test_generator_error_handling(mock_groq_client):
    mock_groq_client.chat.completions.create = AsyncMock(
        side_effect=Exception("Groq rate limit exceeded")
    )

    generator = LLMGenerator(api_key="mock_key")
    with pytest.raises(GenerationError):
        await generator.generate(prompt="Hello")