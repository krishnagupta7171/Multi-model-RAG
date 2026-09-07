import pytest
from src.generation.prompt_temp import (PromptTemplate,PromptTemplates,format_rag_prompt,get_system_prompt,)


def test_prompt_template_dynamic_format():
    tpl = PromptTemplate(
        template="Hello {name}, welcome to {domain}!",
        input_variables=["name", "domain"],
    )
    result = tpl.format(name="User", domain="RAG")
    assert result == "Hello User, welcome to RAG!"


def test_prompt_template_missing_var_raises():
    tpl = PromptTemplate(
        template="{var_a} and {var_b}",
        input_variables=["var_a", "var_b"],
    )
    with pytest.raises(ValueError) as exc:
        tpl.format(var_a="present")
    assert "Missing required prompt variables: var_b" in str(exc.value)


def test_rag_prompt_citations():
    query = "What is RAG?"
    context = ["Chunk 1 content", "Chunk 2 content"]
    prompt = PromptTemplates.rag_prompt(query, context, include_sources=True)

    assert "[Document 1]\nChunk 1 content" in prompt
    assert "[Document 2]\nChunk 2 content" in prompt
    assert "Question: What is RAG?" in prompt
    assert "cite the document numbers" in prompt


def test_rag_prompt_no_citations():
    query = "What is RAG?"
    context = ["Chunk 1 content"]
    prompt = PromptTemplates.rag_prompt(query, context, include_sources=False)

    assert "cite the document numbers" not in prompt


def test_query_rewrite_and_multi_query():
    rewrite = PromptTemplates.query_rewrite_prompt("vector db search")
    multi = PromptTemplates.multi_query_prompt("vector db search")

    assert "Original query: vector db search" in rewrite
    assert "Question: vector db search" in multi


def test_convenience_functions():
    prompt = format_rag_prompt("hello", ["world"])
    system = get_system_prompt()

    assert "Question: hello" in prompt
    assert "You are a knowledgeable AI assistant" in system