from typing import Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Okapi

from ..observability.logging import get_logger
from ..retrieval.embeddings import EmbeddingService, get_embedding_service
from ..retrieval.vector_store import VectorStoreRetriever
from ..utils.config import get_settings
from ..utils.exceptions import RetrievalError

logger = get_logger(__name__)


class BM25Retriever:
    # BM25 retriever for sparse retrieval using the rank_bm25 library.
    def __init__(self):
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_documents(self,documents: List[Tuple[str, str]],) -> None:
#        Index documents for BM25 retrieval.
        logger.info(f"Indexing {len(documents)} documents for BM25")

        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.corpus = [text for _, text in documents]

        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 indexing complete")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # Search the BM25 index for the given query. and return the top_k results as a list of (doc_id, score) tuples.
        # raise RetrievalError if the BM25 index is not initialized. Log the number of results found.
        if self.bm25 is None or not self.doc_ids:
            raise RetrievalError("BM25 index not initialized. Call index_documents first.")

        if not query.strip():
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k results
            top_indices = sorted(range(len(scores)),key=lambda i: scores[i],reverse=True,)[:top_k]

            results = [(self.doc_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
]

            logger.debug(f"BM25 found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise RetrievalError("BM25 search failed", original_error=e)


class HybridRetriever:
   # Hybrid retriever that combines dense vector retrieval and sparse
   #  BM25 retrieval using reciprocal rank fusion.
    def __init__(self,vector_retriever: VectorStoreRetriever,
        embedding_service: Optional[EmbeddingService] = None,alpha: float = 0.5,):
        # Initialize the hybrid retriever with a vector retriever, 
        # optional embedding service, and alpha weight for dense retrieval.
        self.vector_retriever = vector_retriever
        self.embedding_service = embedding_service or get_embedding_service()
        self.bm25_retriever = BM25Retriever()
        self.alpha = alpha  # Dense weight
        self.settings = get_settings()

    async def index_documents(self,documents: List[Tuple[str, str]],) -> None:
       # Index documents for both BM25 and vector retrieval.
       #  The documents should be a list of (doc_id, text) tuples.
       #  Log the number of documents indexed.
        logger.info(f"Indexing {len(documents)} documents for hybrid search")

        # Index for BM25 (sparse)
        self.bm25_retriever.index_documents(documents)

        # Index for vector search (dense)
        texts = [text for _, text in documents]
        ids = [doc_id for doc_id, _ in documents]
        await self.vector_retriever.add_texts(texts=texts, ids=ids)

        logger.info("Hybrid indexing complete")

    async def search(self,query: str,
        top_k: Optional[int] = None,alpha: Optional[float] = None,) -> List[Tuple[str, float, str]]:
        # Perform hybrid search using both dense and sparse retrieval.
        #  The results are combined using reciprocal rank fusion.

        try:
            top_k = top_k or getattr(self.settings, "retrieval_top_k", 5)
            alpha = alpha if alpha is not None else self.alpha

            logger.debug(f"Hybrid search with alpha={alpha}")

            # Dense retrieval
            dense_results = await self.vector_retriever.similarity_search(query=query,top_k=top_k * 2,)  # Get more for fusion

            # Sparse retrieval
            sparse_results = self.bm25_retriever.search(query=query,top_k=top_k * 2,)

            # Reciprocal rank fusion
            fused_results = self._reciprocal_rank_fusion(dense_results=dense_results,
                sparse_results=sparse_results,alpha=alpha,top_k=top_k,)

            logger.info(f"Hybrid search returned {len(fused_results)} results")
            return fused_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError("Hybrid search failed", original_error=e)

    def _reciprocal_rank_fusion(self,dense_results: List[Tuple[str, float, str]],
        sparse_results: List[Tuple[str, float]],alpha: float,top_k: int,k: int = 60,) -> List[Tuple[str, float, str]]:
        # Perform reciprocal rank fusion of dense and sparse results.
        #  dense_results: List of (doc_id, score, text) from dense retrieval
        doc_map: Dict[str, str] = {
            doc_id: text for doc_id, _, text in dense_results
        }

        # Also incorporate texts from BM25 corpus if missing from dense results
        for doc_id, text in zip(self.bm25_retriever.doc_ids, self.bm25_retriever.corpus):
            if doc_id not in doc_map:
                doc_map[doc_id] = text

        rrf_scores: Dict[str, float] = {}

        # Dense scores (weighted by alpha)
        for rank, (doc_id, _, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + alpha / (k + rank)

        # Sparse scores (weighted by 1-alpha)
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1 - alpha) / (k + rank)

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(),key=lambda x: x[1],reverse=True,)[:top_k]

        results = [(doc_id, score, doc_map.get(doc_id, ""))
            for doc_id, score in sorted_docs
            if doc_id in doc_map
        ]

        return results


class MultiQueryRetriever:
# Multi-query retriever that generates multiple queries from the original query
#  using an LLM and retrieves results for each query.
    def __init__(self,base_retriever: VectorStoreRetriever,llm_client=None,):
        # Initialize the multi-query retriever with a base retriever and 
        # an optional LLM client for query generation.
        self.base_retriever = base_retriever
        self.llm_client = llm_client

    async def search(self,query: str,num_queries: int = 3,top_k: int = 10,) -> List[Tuple[str, float, str]]:
        # Generate multiple queries from the original query and retrieve results for each.
        queries = await self._generate_queries(query, num_queries)

        all_results: List[Tuple[str, float, str]] = []
        seen_ids: Set[str] = set()

        for q in queries:
            results = await self.base_retriever.similarity_search(query=q,top_k=top_k,)

            for doc_id, score, text in results:
                if doc_id not in seen_ids:
                    all_results.append((doc_id, score, text))
                    seen_ids.add(doc_id)

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    async def _generate_queries(self,query: str,num_queries: int,) -> List[str]:
        # Generate multiple queries from the original query using the LLM client.
        #  If the LLM client is not provided, return the original query in a list.
        if not query or not query.strip():   
            return [query]
        if self.llm_client is None:
            return [query]

        from ..generation.prompt_temp import PromptTemplates

        prompt = PromptTemplates.multi_query_prompt(query)

        try:
            response = await self.llm_client.generate(prompt=prompt,temperature=0.3,max_tokens=200,)

            generated = [q.strip() for q in response.split("\n") if q.strip()]
            queries = [query] + generated[:num_queries - 1]

            logger.debug(f"Generated {len(queries)} queries")
            return queries

        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return [query]