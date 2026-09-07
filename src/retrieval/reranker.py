from typing import List, Tuple

from ..observability.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


class ReRanker:

    async def rerank(self,query: str,
        documents: List[Tuple[str, float, str]],top_k: int = 5,) -> List[Tuple[str, float, str]]:
#  Re-rank documents based on the query and return the top-k results.
        raise NotImplementedError


class SimpleScoreReRanker(ReRanker):
    #Simple re-ranker that sorts documents by their existing scores.

    async def rerank(self,query: str,
        documents: List[Tuple[str, float, str]],top_k: int = 5,) -> List[Tuple[str, float, str]]:
        #Re-rank documents by their existing scores and return the top-k results.
        if not documents:
            return []
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class KeywordReRanker(ReRanker):
    #Re-ranker that boosts scores based on keyword matching between the query and document text.
    def __init__(self, keyword_weight: float = 0.3):
        #Initialize the keyword re-ranker with a specified weight for keyword matching.
        self.keyword_weight = keyword_weight

    async def rerank(self,query: str,
        documents: List[Tuple[str, float, str]],top_k: int = 5,) -> List[Tuple[str, float, str]]:
        #Re-rank documents by boosting scores based on keyword matches with the query.
        if not documents:
            return []
        query_terms = set(query.lower().split())

        reranked = []
        for doc_id, score, text in documents:
            text_lower = text.lower()
            matches = sum(1 for term in query_terms if term in text_lower)

            keyword_boost = matches / len(query_terms) if query_terms else 0
            new_score = score + (keyword_boost * self.keyword_weight)

            reranked.append((doc_id, new_score, text))

        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(documents)} documents with keyword boosting")
        return reranked[:top_k]


class LengthNormalizedReRanker(ReRanker):
    #Re-ranker that normalizes scores based on document length to avoid bias towards longer documents.
    def __init__(self, length_penalty: float = 0.1):
        #Initialize the length-normalized re-ranker with a specified penalty for length normalization.
        self.length_penalty = length_penalty

    async def rerank(self,query: str,
        documents: List[Tuple[str, float, str]],top_k: int = 5,) -> List[Tuple[str, float, str]]:
        #Re-rank documents by normalizing scores based on their length relative to the average document length.
        if not documents:
            return []

        avg_length = sum(len(text) for _, _, text in documents) / len(documents)

        reranked = []
        for doc_id, score, text in documents:
            length_ratio = len(text) / avg_length if avg_length > 0 else 1
            length_factor = 1 / (1 + self.length_penalty * abs(length_ratio - 1))

            new_score = score * length_factor
            reranked.append((doc_id, new_score, text))

        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(documents)} documents with length normalization")
        return reranked[:top_k]


class CombinedReRanker(ReRanker):
    #Combined re-ranker that applies multiple re-ranking strategies sequentially.

    def __init__(self,use_keywords: bool = True,use_length_norm: bool = True,
        keyword_weight: float = 0.3,length_penalty: float = 0.1,):
        #Initialize the combined re-ranker with options to use keyword boosting and length normalization.
        self.rerankers = []

        if use_keywords:
            self.rerankers.append(KeywordReRanker(keyword_weight=keyword_weight))

        if use_length_norm:
            self.rerankers.append(
                LengthNormalizedReRanker(length_penalty=length_penalty)
            )

    async def rerank(self,query: str,documents: List[Tuple[str, float, str]],
        top_k: int = 5,) -> List[Tuple[str, float, str]]:
       #  Re-rank documents by applying multiple re-ranking strategies sequentially and return the top-k results.
        if not documents:
            return []
        current_docs = documents

        for reranker in self.rerankers:
            current_docs = await reranker.rerank(
                query=query,documents=current_docs,top_k=len(current_docs),
            )

        logger.debug(f"Applied {len(self.rerankers)} re-ranking strategies")
        return current_docs[:top_k]


def get_reranker(strategy: str = "combined") -> ReRanker:
   #  Factory function to get a re-ranker instance based on the specified strategy.
    settings = get_settings()

    if strategy == "simple":
        return SimpleScoreReRanker()
    elif strategy == "keyword":
        return KeywordReRanker()
    elif strategy == "length":
        return LengthNormalizedReRanker()
    elif strategy == "combined":
        return CombinedReRanker()
    else:
        logger.warning(f"Unknown re-ranker strategy: {strategy}, using simple")
        return SimpleScoreReRanker()