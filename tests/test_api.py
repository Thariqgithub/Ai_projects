import pytest
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.collection_stats.return_value = {"total_chunks": 42, "sources": ["doc.pdf"]}
    pipeline.ingest_text.return_value = 5
    pipeline.query.return_value = {
        "answer": "Test answer",
        "sources": [
            {"id": "1", "source": "doc.pdf", "filename": "doc.pdf", "score": 0.9}
        ],
        "chunks": [{"id": "1", "content": "chunk text", "metadata": {}}],
    }
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    with patch("api.routes.get_pipeline", return_value=mock_pipeline):
        from api.main import app
        return TestClient(app)


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["total_chunks"] == 42


def test_ingest_text(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "This is a test document with enough content to ingest.", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_stored"] == 5


def test_query(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert len(data["sources"]) == 1
    assert data["question"] == "What is RAG?"


def test_collection_stats(client):
    response = client.get("/api/v1/collection/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 42
    assert "doc.pdf" in data["sources"]


def test_delete_source(client, mock_pipeline):
    mock_pipeline.delete_source.return_value = None
    response = client.request(
        "DELETE",
        "/api/v1/collection/source",
        json={"source": "doc.pdf"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
