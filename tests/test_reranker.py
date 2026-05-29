import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.reranker import SimpleReranker


@pytest.fixture
def sample_chunks():
    return [
        {"id": "1", "content": "Python is a popular programming language.", "metadata": {}, "score": 0.7},
        {"id": "2", "content": "The sky is blue on a clear day.", "metadata": {}, "score": 0.5},
        {"id": "3", "content": "Python supports object-oriented programming.", "metadata": {}, "score": 0.65},
        {"id": "4", "content": "Machine learning is a subset of AI.", "metadata": {}, "score": 0.6},
        {"id": "5", "content": "Python is used for data science and machine learning.", "metadata": {}, "score": 0.8},
    ]


def test_simple_reranker_returns_top_n(sample_chunks):
    reranker = SimpleReranker()
    results = reranker.rerank("Python programming", sample_chunks, top_n=3)
    assert len(results) == 3


def test_simple_reranker_orders_by_relevance(sample_chunks):
    reranker = SimpleReranker()
    results = reranker.rerank("Python programming language", sample_chunks, top_n=5)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_simple_reranker_adds_score_field(sample_chunks):
    reranker = SimpleReranker()
    results = reranker.rerank("Python", sample_chunks)
    for r in results:
        assert "rerank_score" in r


def test_simple_reranker_empty_input():
    reranker = SimpleReranker()
    results = reranker.rerank("query", [])
    assert results == []


def test_simple_reranker_irrelevant_query(sample_chunks):
    reranker = SimpleReranker()
    results = reranker.rerank("quantum physics neutron stars", sample_chunks, top_n=3)
    # Scores should be low but results still returned
    assert len(results) == 3
