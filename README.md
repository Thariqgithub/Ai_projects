# Multi-Model RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system that supports **Gemini**, **OpenAI GPT**, and **Anthropic Claude** as LLM backends, with hybrid retrieval, reranking, background workers, and RAGAS evaluation.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (api/)
    │
    ├─► Ingest Pipeline
    │     loader → preprocessor → chunker → embedder → ChromaDB
    │
    └─► Query Pipeline
          Hybrid Retrieval (Dense + BM25)
              │
          Reranker (Cohere / SimpleReranker)
              │
          PromptBuilder
              │
          LLM Router → Gemini / OpenAI / Anthropic
              │
          Streamed or Synchronous Response
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env .env.local
# Edit .env with your API keys
```

Required keys:
- `GOOGLE_API_KEY` — Gemini embeddings + generation
- `OPENAI_API_KEY` — GPT-4o generation (optional)
- `ANTHROPIC_API_KEY` — Claude generation (optional)
- `COHERE_API_KEY` — Cohere reranker (optional, falls back to SimpleReranker)

### 3. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 4. Ingest documents

```bash
# Via CLI
python scripts/ingest_documents.py --path ./data/documents

# Via API
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@report.pdf"
```

### 5. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5, "llm_model": "gemini"}'
```

---

## Docker

```bash
cd docker
docker compose up --build

# With Celery monitoring (Flower at :5555)
docker compose --profile monitoring up --build
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/ingest/file` | Upload & ingest a file |
| POST | `/api/v1/ingest/text` | Ingest raw text |
| POST | `/api/v1/ingest/directory` | Ingest a directory |
| POST | `/api/v1/query` | RAG query (sync) |
| POST | `/api/v1/query/stream` | RAG query (SSE stream) |
| GET | `/api/v1/collection/stats` | Collection statistics |
| DELETE | `/api/v1/collection/source` | Delete a source from index |

---

## LLM Routing

The `LLMRouter` auto-routes queries by heuristics:

| Keyword Pattern | Routed To |
|-----------------|-----------|
| code, python, debug, algorithm | Anthropic Claude |
| analyze, compare, summarize, table | OpenAI GPT |
| Everything else | Gemini |

Override per-request: `"llm_model": "openai"` in the query body.

---

## Background Tasks (Celery)

Start worker:
```bash
celery -A workers.celery_worker worker --loglevel=info
```

Use tasks programmatically:
```python
from workers.tasks import ingest_file_task, rag_query_task

# Fire and forget
result = ingest_file_task.delay("./data/documents/report.pdf")
print(result.get())  # {"status": "success", "chunks": 42}

# Async query
result = rag_query_task.delay("What is the capital of France?")
print(result.get())  # {"answer": "Paris", ...}
```

---

## Evaluation (RAGAS)

```python
from rag.rag_pipeline import RAGPipeline
from evaluation.ragas_eval import RAGASEvaluator

pipeline = RAGPipeline()
evaluator = RAGASEvaluator()

test_cases = [
    {"question": "What is ChromaDB?", "ground_truth": "A vector database."},
    {"question": "What is RAG?", "ground_truth": "Retrieval-augmented generation."},
]

scores = evaluator.evaluate_pipeline(pipeline, test_cases)
print(scores)
# {
#   "faithfulness": 0.91,
#   "answer_relevancy": 0.87,
#   "context_precision": 0.83,
#   "context_recall": 0.79,
# }
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Rebuild Index

```bash
python scripts/rebuild_index.py --dir ./data/documents --confirm
```

---

## Project Structure

```
multi_model_rag/
├── api/            FastAPI routes, schemas, app entry point
├── ingestion/      Loader, preprocessor, chunker
├── embeddings/     Gemini embedding model
├── vectorstore/    ChromaDB client
├── retrieval/      Dense, hybrid, reranker
├── router/         Multi-model LLM routing
├── rag/            Prompt builder, full pipeline
├── workers/        Celery tasks and worker
├── evaluation/     RAGAS evaluation
├── utils/          Logger, helpers
├── config/         Settings (pydantic-settings)
├── scripts/        CLI tools
├── tests/          Unit tests
└── docker/         Dockerfile + docker-compose
```

---

## Supported File Types

| Extension | Loader |
|-----------|--------|
| `.pdf` | PyMuPDF |
| `.docx` | python-docx |
| `.txt` / `.md` / `.csv` | Plain text |
| `.png` / `.jpg` / `.jpeg` | Tesseract OCR |
