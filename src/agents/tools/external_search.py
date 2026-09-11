
import os
from typing import Any, Dict, List, Optional

from ..core_agent import Tool
from ...observability.logging import get_logger
from ...utils.exceptions import AgentError

logger = get_logger(__name__)


class ExternalSearchTool(Tool):
   # Tool for performing web searches using an external search API.
    def __init__(self,api_key: Optional[str] = None,max_results: int = 5,name: str = "web_search",
        description: str = "Search the live web for current information, external facts, and real-time updates.",):
        super().__init__(name=name, description=description)
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self._client: Any = None

    def _get_client(self) -> Any:
        # Lazily initialize the Tavily client if an API key is provided.
        if self._client is None and self.api_key:
            try:
                from tavily import AsyncTavilyClient
                self._client = AsyncTavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("tavily-python not installed. Falling back to placeholder results.")
        return self._client

    async def execute(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        # Execute a web search for the given query using the Tavily API 
        # or return placeholder results if no API key is configured.
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if not query.strip():
            logger.warning("Empty query passed to web search tool")
            return []

        limit = max_results or self.max_results
        logger.info(f"Executing web search: '{query}' (limit={limit})")

        client = self._get_client()

        # If a live client is configured, execute real API search
        if client is not None:
            try:
                response = await client.search(query=query.strip(),max_results=limit,search_depth="basic",)
                raw_results = response.get("results", [])
                return [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in raw_results
                ]
            except Exception as e:
                logger.error(f"Live external search failed for query '{query}': {e}")
                raise AgentError(f"External search failed for query: '{query}'", original_error=e) from e

        # Fallback placeholder mode for local/offline testing
        logger.warning("Running web search in mock/placeholder mode (no API key configured)")
        return [
            {
                "title": f"Search result {i + 1} for: {query.strip()}",
                "url": f"https://example.com/result{i + 1}",
                "content": f"Placeholder snippet for query: {query.strip()}",
                "score": round(1.0 - (i * 0.1), 2),
            }
            for i in range(limit)
        ]