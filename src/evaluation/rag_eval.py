from typing import Any, Dict, List, Optional, Set

from ..observability.logging import get_logger

logger = get_logger(__name__)


def compute_answer_relevancy(query: str, answer: str, expected_keywords: List[str]) -> float:
    # Compute the relevancy of the answer based on the presence of expected keywords.
    # The relevancy score is calculated as the ratio of matched keywords to the total number of
    # expected keywords, resulting in a score between 0.0 and 1.0.
    # If the answer is empty or None, the relevancy score is 0.0
    if not answer or not answer.strip():
        return 0.0
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return float(matches / len(expected_keywords))


def compute_context_precision(retrieved_doc_ids: List[str],ground_truth_doc_ids: List[str],) -> float:
    # Calculate context precision (true positives retrieved / total retrieved).
    # The precision score is calculated as the ratio of correctly retrieved document IDs to the total number
             # of retrieved document IDs, resulting in a score between 0.0 and 1.0.
    # If no documents were retrieved, the precision score is 0.0.

    if not retrieved_doc_ids:
        return 0.0

    retrieved_set: Set[str] = set(retrieved_doc_ids)
    relevant_set: Set[str] = set(ground_truth_doc_ids)

    true_positives = len(retrieved_set & relevant_set)
    return float(true_positives / len(retrieved_set))


def compute_context_recall(retrieved_doc_ids: List[str],ground_truth_doc_ids: List[str],) -> float:
    # Calculate context recall (true positives retrieved / total relevant).
    # The recall score is calculated as the ratio of correctly retrieved document IDs to the total number
            # of relevant document IDs, resulting in a score between 0.0 and 1.0.
    # If there are no relevant documents, the recall score is defined as 1.0
    if not ground_truth_doc_ids:
        return 1.0

    retrieved_set: Set[str] = set(retrieved_doc_ids)
    relevant_set: Set[str] = set(ground_truth_doc_ids)

    true_positives = len(retrieved_set & relevant_set)
    return float(true_positives / len(relevant_set))


def compute_token_overlap_similarity(candidate_answer: str, reference_answer: str) -> float:
    # Compute the Jaccard similarity between the candidate answer and the reference answer based on token overlap.
    # The similarity score is calculated as the size of the intersection of unique tokens divided by the
             # size of the union of unique tokens, resulting in a score between 0.0 and 1.0.
    # If either answer is empty or None, the similarity score is 0.0
    words_candidate = set(candidate_answer.lower().split())
    words_reference = set(reference_answer.lower().split())

    if not words_candidate or not words_reference:
        return 0.0

    intersection = words_candidate & words_reference
    union = words_candidate | words_reference
    return float(len(intersection) / len(union))


class RAGMetricsTracker:
    # Class to track and compute aggregate metrics for RAG evaluation across multiple queries.
    def __init__(self):
        self.total_queries: int = 0
        self.total_docs_retrieved: int = 0
        self.reflection_scores: List[float] = []
        self.relevancy_scores: List[float] = []
        self.precision_scores: List[float] = []
        self.recall_scores: List[float] = []

    def record_run(self,num_docs: int,reflection_score: Optional[float] = None,relevancy: Optional[float] = None,
        precision: Optional[float] = None,recall: Optional[float] = None,) -> None:
        # Record the metrics for a single query run, updating the total counts and lists of scores.
        # The method increments the total number of queries and documents retrieved, and appends the provided scores to their respective lists if they are not None.
        # This allows for the computation of aggregate averages and summary statistics across multiple runs.
        # The method is designed to be called after each query evaluation to maintain an ongoing record of performance metrics.
        
        self.total_queries += 1
        self.total_docs_retrieved += max(0, num_docs)

        if reflection_score is not None:
            self.reflection_scores.append(float(reflection_score))
        if relevancy is not None:
            self.relevancy_scores.append(float(relevancy))
        if precision is not None:
            self.precision_scores.append(float(precision))
        if recall is not None:
            self.recall_scores.append(float(recall))

    def get_summary(self) -> Dict[str, Any]:
        # Compute and return a summary of the tracked metrics, including total queries, average documents retrieved per query,
        # average reflection score, average relevancy, average precision, and average recall.
        def safe_avg(lst: List[float]) -> float:
            return float(sum(lst) / len(lst)) if lst else 0.0

        return {
            "total_queries": self.total_queries,
            "avg_docs_per_query": (
                float(self.total_docs_retrieved / self.total_queries)
                if self.total_queries > 0
                else 0.0
            ),
            "avg_reflection_score": safe_avg(self.reflection_scores),
            "avg_relevancy": safe_avg(self.relevancy_scores),
            "avg_precision": safe_avg(self.precision_scores),
            "avg_recall": safe_avg(self.recall_scores),
        }