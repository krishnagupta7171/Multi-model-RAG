from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import DatabaseError


logger = get_logger(__name__)


class VectorStore(ABC):
    #Abstract interface for vector stores.

    @abstractmethod
    async def create_collection(self,name: str,dimension: int,) -> None:
        #Create a vector collection.
        pass

    @abstractmethod
    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        #Add documents and their embeddings to a collection.
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        #Search for similar documents.
        pass

    @abstractmethod
    async def delete_documents(self,collection_name: str,ids: List[str],) -> None:
        #Delete documents from a collection.
        pass


class ChromaVectorStore(VectorStore):
    #ChromaDB implementation of the vector store.

    def __init__(self,persist_directory: Optional[str] = None,):
        settings = get_settings()

        self.persist_directory = (
            persist_directory
            if persist_directory is not None
            else settings.chroma_persist_dir
        )

        logger.info(
            f"Initializing ChromaDB at {self.persist_directory}"
        )

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
            ),
        )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise DatabaseError("Failed to initialize ChromaDB",
                original_error=e,
            )

    async def create_collection(self,name: str,dimension: int,) -> None:
        #Create a collection if it does not already exist.
        try:
            self.client.get_or_create_collection(name=name)

            logger.info(
                f"Created or retrieved ChromaDB collection: {name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create collection {name}: {e}"
            )
            raise DatabaseError(
                f"Failed to create collection {name}",
                original_error=e,
            )

    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        #Add documents and embeddings to ChromaDB.
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name
            )

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )

            logger.info(
                f"Added {len(ids)} documents to {collection_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to add documents to {collection_name}: {e}"
            )
            raise DatabaseError(
                f"Failed to add documents to {collection_name}",
                original_error=e,
            )

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        #Search ChromaDB for similar documents.
        try:
            collection = self.client.get_collection(
                name=collection_name
            )

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
            )

            formatted_results = []

            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        (
                            results["ids"][0][i],
                            results["distances"][0][i],
                            {
                                "text": results["documents"][0][i],
                                "metadata": (
                                    results["metadatas"][0][i]
                                    if results["metadatas"]
                                    else {}
                                ),
                            },
                        )
                    )

            logger.debug(
                f"Found {len(formatted_results)} results"
            )

            return formatted_results

        except Exception as e:
            logger.error(
                f"Search failed in {collection_name}: {e}"
            )
            raise DatabaseError(
                f"Search failed in {collection_name}",
                original_error=e,
            )

    async def delete_documents(self,collection_name: str,ids: List[str],) -> None:
        #Delete documents from ChromaDB.
        try:
            collection = self.client.get_collection(
                name=collection_name
            )

            collection.delete(ids=ids)

            logger.info(
                f"Deleted {len(ids)} documents from {collection_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to delete documents: {e}"
            )
            raise DatabaseError(
                f"Failed to delete documents from {collection_name}",
                original_error=e,
            )


def get_vector_store(store_type: Optional[str] = None,) -> VectorStore:
    
    settings = get_settings()

    store_type = store_type or settings.vector_db_type

    if store_type.lower() == "chroma":
        return ChromaVectorStore()

    raise DatabaseError(
        f"Unknown vector store type: {store_type}"
    )
