import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.prompt_builder import PromptBuilder


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "abc1",
            "content": "ChromaDB is a vector database for storing embeddings.",
            "metadata": {"source": "docs/chroma.txt", "filename": "chroma.txt"},
            "score": 0.92,
        },
        {
            "id": "abc2",
            "content": "FAISS is another popular vector store for dense retrieval.",
            "metadata": {"source": "docs/faiss.txt", "filename": "faiss.txt"},
            "score": 0.85,
        },
    ]


def test_build_returns_system_and_user(sample_chunks):
    builder = PromptBuilder()
    prompt = builder.build("What is ChromaDB?", sample_chunks)
    assert "system" in prompt
    assert "user" in prompt


def test_context_contains_chunk_content(sample_chunks):
    builder = PromptBuilder()
    prompt = builder.build("What is ChromaDB?", sample_chunks)
    assert "ChromaDB" in prompt["user"]
    assert "FAISS" in prompt["user"]


def test_context_includes_source(sample_chunks):
    builder = PromptBuilder(include_metadata=True)
    prompt = builder.build("query", sample_chunks)
    assert "chroma.txt" in prompt["user"]


def test_max_context_tokens():
    builder = PromptBuilder(max_context_tokens=10)  # Very small budget
    big_chunks = [
        {"id": str(i), "content": "x" * 500, "metadata": {}, "score": 0.9}
        for i in range(20)
    ]
    prompt = builder.build("query", big_chunks)
    # Should not include all 20 chunks
    assert prompt["user"].count("[Chunk") < 20


def test_empty_chunks():
    builder = PromptBuilder()
    prompt = builder.build("query", [])
    assert "Answer:" in prompt["user"]


def test_build_conversation(sample_chunks):
    builder = PromptBuilder()
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    messages = builder.build_conversation("Follow-up question", sample_chunks, history)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"
