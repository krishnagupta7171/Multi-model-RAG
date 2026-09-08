from typing import List, Optional
import re
from ..observability.logging import get_logger
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class QueryTransformer:
    # Query transformer for rewriting, expanding, and adding context to queries.

    def __init__(self, llm_client=None):
        # Initialize the llm client for query transformation.
        self.llm_client = llm_client

    async def rewrite_query(self, query: str) -> str:
        
        # Rewrite the query to improve retrieval performance.
        if not query:
            return query
        if self.llm_client is None:
            return query

        from ..generation.prompt_temp import PromptTemplates

        prompt = PromptTemplates.query_rewrite_prompt(query)

        try:
            response = await self.llm_client.generate(prompt=prompt,temperature=0.3,max_tokens=150,)

            # Take the first rewritten query
            rewritten = response.split("\n")[0].strip()
            logger.debug(f"Rewritten query: {query} -> {rewritten}")

            return rewritten if rewritten else query

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query

    async def expand_query(self, query: str) -> List[str]:
        # Expand the query into multiple variations for broader retrieval. return multiple variations of the query in a list. If expansion fails, return the original query in a list.
        if not query:
            return [query]
        if self.llm_client is None:
            return [query]

        from ..generation.prompt_temp import PromptTemplates

        prompt = PromptTemplates.multi_query_prompt(query)

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            # Parse variations
            variations = [q.strip() for q in response.split("\n") if q.strip()]
            all_queries = [query] + variations

            logger.debug(f"Expanded query into {len(all_queries)} variations")
            return all_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def add_context(self, query: str, context: str) -> str:
        # Add context to the query for better retrieval. If context is empty, return the original query.
        if not context:
            return query

        return f"Context: {context}\n\nQuestion: {query}"

    def extract_keywords(self, query: str) -> List[str]:
        # Extract keywords from the query for keyword-based retrieval. If query is empty, return an empty list.
        if not query:
            return []
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "what",
            "when",
            "where",
            "who",
            "how",
        }

        cleaned_query = re.sub(r"[^\w\s]", "", query.lower())
        words = cleaned_query.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords


class HyDETransformer:
    # Generate hypothetical documents to improve retrieval for complex queries.

    def __init__(self, llm_client):
        # Initialize the llm client for HyDE generation.
        self.llm_client = llm_client

    async def generate_hypothetical_document(self, query: str) -> str:
       # Generate a hypothetical document based on the query to improve retrieval. If generation fails, raise a RetrievalError.
        if not query:   
            return ""
        prompt = f"""Generate a detailed passage that would answer the following question.
Write as if you are an expert providing a comprehensive answer.

Question: {query}

Hypothetical passage:"""

        try:
            response = await self.llm_client.generate(prompt=prompt,temperature=0.4,max_tokens=300,)

            logger.debug("Generated hypothetical document for query")
            return response.strip()

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            raise RetrievalError(
                "Failed to generate hypothetical document",
                original_error=e,
            )


class QueryDecomposer:
    # Decompose complex queries into simpler sub-queries for better retrieval.

    def __init__(self, llm_client):
        # Initialize the llm client for query decomposition.
        self.llm_client = llm_client

    async def decompose(self, query: str) -> List[str]:
        # Decompose the query into simpler sub-queries. If decomposition fails, return the original query in a list.
        if not query:
            return [query]
        prompt = f"""Break down the following complex question into 2-4 simpler sub-questions.
Each sub-question should focus on one specific aspect.

Complex question: {query}

Sub-questions (one per line):"""

        try:
            response = await self.llm_client.generate(prompt=prompt,temperature=0.3,max_tokens=200,)

            # Parse sub-queries
            sub_queries = [q.strip() for q in response.split("\n") if q.strip()]

            logger.debug(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries if sub_queries else [query]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]


# Global transformer instance
_query_transformer: Optional[QueryTransformer] = None


def get_query_transformer(llm_client=None) -> QueryTransformer:
# Get a singleton instance of QueryTransformer. If it doesn't exist, create one with the provided llm_client.   
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = QueryTransformer(llm_client=llm_client)
    return _query_transformer