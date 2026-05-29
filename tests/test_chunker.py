import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import TextChunker


@pytest.fixture
def sample_doc():
    return {
        "content": (
            "Artificial intelligence is transforming every industry. "
            "Machine learning models can now process images, text, and audio. "
            "Deep learning has enabled breakthroughs in natural language processing. "
            "Transformer architectures power modern LLMs like GPT and Gemini. "
            "Retrieval-augmented generation combines search with generation. "
            "Vector databases store high-dimensional embeddings efficiently. "
            "Semantic search outperforms keyword search for complex queries. "
            "Hybrid retrieval fuses dense and sparse signals for better recall. "
            "Reranking improves precision by reordering retrieved chunks. "
            "Evaluation frameworks like RAGAS measure RAG pipeline quality."
        ) * 5,
        "metadata": {"source": "test.txt", "filename": "test.txt"},
    }


def test_fixed_chunking(sample_doc):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, strategy="fixed")
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) > 1
    for c in chunks:
        assert "id" in c
        assert "content" in c
        assert "metadata" in c


def test_recursive_chunking(sample_doc):
    chunker = TextChunker(chunk_size=300, chunk_overlap=30, strategy="recursive")
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) > 0
    assert all(len(c["content"]) > 0 for c in chunks)


def test_sentence_chunking(sample_doc):
    chunker = TextChunker(chunk_size=250, chunk_overlap=25, strategy="sentence")
    chunks = chunker.chunk(sample_doc)
    assert len(chunks) > 0


def test_chunk_metadata(sample_doc):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_doc)
    for c in chunks:
        assert c["metadata"]["chunk_total"] == len(chunks)
        assert "chunk_index" in c["metadata"]


def test_empty_document():
    chunker = TextChunker()
    chunks = chunker.chunk({"content": "   ", "metadata": {}})
    assert chunks == []


def test_chunk_many():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    docs = [
        {"content": "Document one " * 20, "metadata": {"source": "a.txt"}},
        {"content": "Document two " * 20, "metadata": {"source": "b.txt"}},
    ]
    chunks = chunker.chunk_many(docs)
    assert len(chunks) > 2
