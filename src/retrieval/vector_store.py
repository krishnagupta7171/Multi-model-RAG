import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..database.vector_database import VectorStore, get_vector_store
from ..observability.logging import get_logger
from ..retrieval.embeddings import EmbeddingService, get_embedding_service
from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class VectorStoreRetriever:

    def __init__(
        self,
        collection_name: str = "documents",
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.collection_name = collection_name
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.settings = get_settings()

    async def initialize(self) -> None:
        await self.vector_store.create_collection(
            name=self.collection_name,
            dimension=self.settings.embedding_dimension,
        )
        logger.info(f"Initialized collection: {self.collection_name}")

    async def add_texts(self,texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,) -> List[str]:

        
        if not texts:
            return []

        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Sanitize metadata to prevent ChromaDB empty-dict errors
            if metadata:
                sanitized_metadata = [
                    meta if meta else {"source": "unknown"} for meta in metadata
                ]
            else:
                sanitized_metadata = [{"source": "unknown"} for _ in texts]

            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = await self.embedding_service.embed_batch_async(texts)

            await self.vector_store.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadata=sanitized_metadata,
            )

            logger.info(f"Added {len(texts)} documents to '{self.collection_name}'")
            return ids

        except Exception as e:
            logger.error(f"Failed to add texts to collection '{self.collection_name}': {e}")
            raise RetrievalError(
                "Failed to index texts in vector store",
                original_error=e,
            )

    async def similarity_search(self,query: str,top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,) -> List[Tuple[str, float, str]]:
    
        try:
            logger.debug(f"Generating query embedding for: {query}")
            query_embedding = await self.embedding_service.embed_text_async(query)

            results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            formatted_results = [
                (doc_id, score, payload.get("text", ""))
                for doc_id, score, payload in results
            ]

            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RetrievalError(
                "Similarity search failed",
                original_error=e,
            )

    async def delete(self, ids: List[str]) -> None:
        
        try:
            await self.vector_store.delete_documents(
                collection_name=self.collection_name,
                ids=ids,
            )
            logger.info(f"Deleted {len(ids)} documents from '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RetrievalError(
                "Failed to delete documents",
                original_error=e,
            )

    async def get_relevant_documents(self,query: str,top_k: int = 5,) -> List[str]:
        results = await self.similarity_search(query=query, top_k=top_k)
        return [text for _, _, text in results]


# Global retriever singleton
_retriever: Optional[VectorStoreRetriever] = None


async def get_retriever(collection_name: str = "documents") -> VectorStoreRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorStoreRetriever(collection_name=collection_name)
        await _retriever.initialize()
    return _retriever