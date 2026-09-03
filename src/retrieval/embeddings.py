from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError


class EmbeddingService:
    #Service for generating text embeddings.

    def __init__(self,model_name: Optional[str] = None,device: str = "cpu",) -> None:
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.dimension = settings.embedding_dimension

        try:
            self.model = SentenceTransformer(self.model_name,device=device,)
        except Exception as e:
            raise RetrievalError(
                f"Failed to load embedding model {self.model_name}",
                original_error=e,
            )

    def embed_text(self, text: str) -> List[float]:
        #Generate embedding for a single text.

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return embedding.tolist()

        except Exception as e:
            raise RetrievalError(
                "Embedding generation failed",
                details={"text_length": len(text)},
                original_error=e,
            )

    def embed_batch(self,texts: List[str],batch_size: int = 32,show_progress: bool = False,) -> List[List[float]]:
        #Generate embeddings for multiple texts.

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )

            return embeddings.tolist()

        except Exception as e:
            raise RetrievalError(
                "Batch embedding generation failed",
                details={"batch_size": len(texts)},
                original_error=e,
            )

    async def embed_text_async(self,text: str,) -> List[float]:
        #Async wrapper for embed_text.

        return self.embed_text(text)

    async def embed_batch_async(self,texts: List[str],batch_size: int = 32,) -> List[List[float]]:
        #Async wrapper for embed_batch.

        return self.embed_batch(texts,batch_size=batch_size,)

    def similarity(
        self,embedding1: List[float],embedding2: List[float],) -> float:
        #Compute cosine similarity between two embeddings.

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        return float(
            np.dot(vec1, vec2)
            / (
                np.linalg.norm(vec1)
                * np.linalg.norm(vec2)
            )
        )


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    #Get global embedding service instance.

    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service