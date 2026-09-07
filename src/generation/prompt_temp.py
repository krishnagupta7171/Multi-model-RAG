from typing import Any, List


class PromptTemplate:
    # PromptTemplate represents a template for generating prompts with dynamic variables.
    # It allows for the creation of structured prompts that can be filled with
    # specific values at runtime.
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs: Any) -> str:
        missing_vars = [
            var for var in self.input_variables if var not in kwargs
        ]

        if missing_vars:
            raise ValueError(
                f"Missing required prompt variables: {', '.join(missing_vars)}"
            )

        return self.template.format(**kwargs)


class PromptTemplates:
    # Reusable dynamic templates

    RAG_BASE = PromptTemplate(
        template="""You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {query}{citation_instruction}

Answer:""",
        input_variables=["context", "query", "citation_instruction"],
    )

    QUERY_REWRITE = PromptTemplate(
        template="""Rewrite the following query to make it more specific and suitable for document retrieval.
Generate 2-3 alternative phrasings that capture the same intent.

Original query: {query}

Rewritten queries (one per line):""",
        input_variables=["query"],
    )

    MULTI_QUERY = PromptTemplate(
        template="""Generate 3 different search queries that would help answer the following question from different angles.
Each query should focus on a different aspect or perspective.

Question: {query}

Search queries (one per line):""",
        input_variables=["query"],
    )

    REFLECTION = PromptTemplate(
        template="""Evaluate the quality of the following answer given the question and context.

Question: {query}

Context:
{context}

Answer:
{answer}

Please evaluate:
1. Does the answer accurately address the question?
2. Is the answer well-supported by the context?
3. Are there any inconsistencies or hallucinations?
4. What could be improved?

Provide a brief evaluation and a quality score from 1-10:""",
        input_variables=["query", "context", "answer"],
    )

    SUMMARIZATION = PromptTemplate(
        template="""Provide a concise summary of the following text in no more than {max_words} words.
Focus on the key points and main ideas.

Text:
{text}

Summary:""",
        input_variables=["text", "max_words"],
    )

    EXTRACTION = PromptTemplate(
        template="""Extract the following information from the text:

{fields}

Text:
{text}

Extracted information (JSON format):""",
        input_variables=["fields", "text"],
    )

    AGENT_PLANNING = PromptTemplate(
        template="""You are a planning agent. Break down the following task into concrete steps.

Available tools:
{tools}

Task: {task}

Provide a step-by-step plan to accomplish this task:""",
        input_variables=["tools", "task"],
    )

    CODE_ANALYSIS = PromptTemplate(
        template="""Analyze the following code and answer the question.

Code:
{code}

Question: {question}

Analysis:""",
        input_variables=["code", "question"],
    )

    @staticmethod
    def rag_prompt(query: str,context: List[str],include_sources: bool = True,) -> str:
        context_text = "\n\n".join(
            [f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(context)]
        )

        citation_instruction = (
            "\n\nPlease cite the document numbers you use in your answer (e.g., [1], [2])."
            if include_sources
            else ""
        )

        return PromptTemplates.RAG_BASE.format(
            context=context_text,
            query=query,
            citation_instruction=citation_instruction,
        )

    @staticmethod
    def system_prompt() -> str:
        return """You are a knowledgeable AI assistant with expertise in retrieving and synthesizing information from documents.
Your responses should be:
- Accurate and based on the provided context
- Clear and well-structured
- Honest when information is not available
- Professional and helpful"""

    @staticmethod
    def query_rewrite_prompt(query: str) -> str:
        return PromptTemplates.QUERY_REWRITE.format(query=query)

    @staticmethod
    def multi_query_prompt(query: str) -> str:
        return PromptTemplates.MULTI_QUERY.format(query=query)

    @staticmethod
    def reflection_prompt(query: str,answer: str,context: List[str],) -> str:
        context_text = "\n".join(context)

        return PromptTemplates.REFLECTION.format(query=query,answer=answer,context=context_text,)

    @staticmethod
    def summarization_prompt(text: str,max_words: int = 100,) -> str:
        return PromptTemplates.SUMMARIZATION.format(text=text,max_words=max_words,)

    @staticmethod
    def extraction_prompt(text: str,fields: List[str],) -> str:
        fields_text = "\n".join(
            [f"- {field}" for field in fields]
        )

        return PromptTemplates.EXTRACTION.format(text=text,fields=fields_text,)

    @staticmethod
    def agent_planning_prompt(task: str,available_tools: List[str],) -> str:
        tools_text = "\n".join(
            [f"- {tool}" for tool in available_tools]
        )

        return PromptTemplates.AGENT_PLANNING.format(task=task,tools=tools_text,)

    @staticmethod
    def code_analysis_prompt(code: str,question: str,) -> str:
        return PromptTemplates.CODE_ANALYSIS.format(code=code,question=question,)


def format_rag_prompt(query: str,context: List[str],) -> str:
    return PromptTemplates.rag_prompt(query, context)


def get_system_prompt() -> str:
    return PromptTemplates.system_prompt()

