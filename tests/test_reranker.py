import pytest
from src.retrieval.reranker import (SimpleScoreReRanker,KeywordReRanker,
    LengthNormalizedReRanker,CombinedReRanker,get_reranker,)


@pytest.fixture
def sample_documents():
    return [
        ("doc_1", 0.5, "Python is a versatile programming language for web and AI."),
        ("doc_2", 0.9, "Chroma is an open-source AI-native vector database."),
        ("doc_3", 0.6, "FastAPI is a high-performance web framework for Python."),
    ]


@pytest.mark.asyncio
async def test_simple_score_reranker(sample_documents):
    reranker = SimpleScoreReRanker()
    results = await reranker.rerank("any query", sample_documents, top_k=2)

    assert len(results) == 2
    assert results[0][0] == "doc_2"  # 0.9 score
    assert results[1][0] == "doc_3"  # 0.6 score


@pytest.mark.asyncio
async def test_keyword_reranker_boosts_matching_terms(sample_documents):
    reranker = KeywordReRanker(keyword_weight=0.5)
    # doc_3 contains both "FastAPI" and "Python", doc_1 contains "Python", doc_2 contains neither
    results = await reranker.rerank("FastAPI Python", sample_documents, top_k=3)

    assert results[0][0] == "doc_3"  # boosted from 0.6 to 1.1
    assert results[0][1] > 1.0


@pytest.mark.asyncio
async def test_length_normalized_reranker_empty():
    reranker = LengthNormalizedReRanker()
    results = await reranker.rerank("query", [], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_length_normalized_reranker_penalizes_outliers():
    docs = [
        ("short", 0.8, "Short text."),
        ("normal", 0.8, "This is a normal length sentence for testing."),
        ("long", 0.8, "This is a significantly longer text chunk that exceeds the average document length by a wide margin."),
    ]
    reranker = LengthNormalizedReRanker(length_penalty=0.5)
    results = await reranker.rerank("test", docs, top_k=3)

    assert len(results) == 3
    # The doc closest to average length receives the highest score
    assert results[0][0] == "normal"


@pytest.mark.asyncio
async def test_combined_reranker(sample_documents):
    reranker = CombinedReRanker(use_keywords=True, use_length_norm=True)
    results = await reranker.rerank("vector database", sample_documents, top_k=1)

    assert len(results) == 1
    assert results[0][0] == "doc_2"  # doc_2 matches keywords


def test_get_reranker_factory():
    assert isinstance(get_reranker("simple"), SimpleScoreReRanker)
    assert isinstance(get_reranker("keyword"), KeywordReRanker)
    assert isinstance(get_reranker("length"), LengthNormalizedReRanker)
    assert isinstance(get_reranker("combined"), CombinedReRanker)
    assert isinstance(get_reranker("unknown_strategy"), SimpleScoreReRanker)