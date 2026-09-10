from typing import Any, Dict, List, Optional

from ..core_agent import Tool
from ...observability.logging import get_logger

logger = get_logger(__name__)


class KnowledgeBaseTool(Tool):
    # Tool for retrieving relevant documents from a knowledge base using a retriever.

    def __init__(self,retriever: Any,top_k: int = 5,name: str = "knowledge_base_retriever",description: str = "Retrieve relevant documents and factual context from the internal knowledge base given a query.",):
        # Initialize the knowledge base tool with a retriever and configuration.
        # If the retriever is a VectorStoreRetriever, it will use similarity_search; if it's a HybridRetriever, it will use search.
        # it will retrieve up to top_k documents for a given query.
        # It receives a name and description for logging and observability purposes.
        super().__init__(name=name, description=description)
        self.retriever = retriever
        self.top_k = top_k

    async def execute(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
       #  Execute the knowledge base retrieval tool to fetch relevant documents for a given query.
       # It supports both VectorStoreRetriever and HybridRetriever instances, and returns a list of documents with their IDs, scores, and content.
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if not query or not query.strip():
            logger.warning("Empty query provided to knowledge base tool")
            return []

        k = top_k or self.top_k
        logger.info(f"Retrieving up to {k} documents for query: {query}")

        # Support both VectorStoreRetriever (.similarity_search) and HybridRetriever (.search)
        if hasattr(self.retriever, "similarity_search"):
            results = await self.retriever.similarity_search(query=query, top_k=k)
        elif hasattr(self.retriever, "search"):
            results = await self.retriever.search(query=query, top_k=k)
        else:
            raise AttributeError("Retriever instance must implement 'similarity_search' or 'search'")

        documents = [
            {
                "id": doc_id,
                "score": float(score),
                "content": text,
            }
            for doc_id, score, text in results
        ]

        logger.info(f"Retrieved {len(documents)} documents")
        return documents