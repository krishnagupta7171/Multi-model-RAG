import pytest

from src.evaluation.rag_eval import (
    RAGMetricsTracker,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_token_overlap_similarity,
)


def test_answer_relevancy():
    answer = "ChromaDB is a vector database used for embedding search."
    keywords = ["vector", "embedding", "chromadb"]
    score = compute_answer_relevancy("What is ChromaDB?", answer, keywords)
    assert score == 1.0

    partial_score = compute_answer_relevancy("What is ChromaDB?", answer, ["vector", "missing_term"])
    assert partial_score == 0.5

    assert compute_answer_relevancy("Query", "", ["keyword"]) == 0.0
    assert compute_answer_relevancy("Query", answer, []) == 1.0


def test_context_precision():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    relevant = ["doc_1", "doc_3", "doc_9"]

    # 2 out of 4 retrieved docs are relevant -> 0.5
    assert compute_context_precision(retrieved, relevant) == 0.5
    assert compute_context_precision([], relevant) == 0.0
    assert compute_context_precision(retrieved, []) == 0.0


def test_context_recall():
    retrieved = ["doc_1", "doc_2"]
    relevant = ["doc_1", "doc_2", "doc_3", "doc_4"]

    # 2 out of 4 relevant docs retrieved -> 0.5
    assert compute_context_recall(retrieved, relevant) == 0.5
    assert compute_context_recall(retrieved, []) == 1.0
    assert compute_context_recall([], relevant) == 0.0


def test_token_overlap_similarity():
    text1 = "FastAPI web framework"
    text2 = "FastAPI asynchronous web framework"
    # Overlap: {"fastapi", "web", "framework"} (3), Union: {"fastapi", "web", "framework", "asynchronous"} (4)
    assert compute_token_overlap_similarity(text1, text2) == 0.75
    assert compute_token_overlap_similarity("", text2) == 0.0


def test_metrics_tracker_summary():
    tracker = RAGMetricsTracker()

    tracker.record_run(
        num_docs=4,
        reflection_score=8.0,
        relevancy=0.9,
        precision=0.75,
        recall=1.0,
    )
    tracker.record_run(
        num_docs=2,
        reflection_score=6.0,
        relevancy=0.7,
        precision=0.5,
        recall=0.8,
    )

    summary = tracker.get_summary()

    assert summary["total_queries"] == 2
    assert summary["avg_docs_per_query"] == 3.0
    assert summary["avg_reflection_score"] == 7.0
    assert summary["avg_relevancy"] == pytest.approx(0.8)
    assert summary["avg_precision"] == pytest.approx(0.625)
    assert summary["avg_recall"] == pytest.approx(0.9)