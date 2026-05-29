# Gemini Embedding Models: Technical Guide

## Overview

Google's Gemini embedding family provides state-of-the-art text embeddings for semantic search,
retrieval-augmented generation, classification, and clustering tasks. This guide covers the
available models, their specifications, and best practices for integration into RAG pipelines.

---

## Available Models (2025)

### gemini-embedding-exp-03-07 (gemini-embedding-2-preview)

The most capable Gemini embedding model available. Key characteristics:

| Property            | Value                                 |
|---------------------|---------------------------------------|
| Output dimension    | 3072                                  |
| Max input tokens    | 8192                                  |
| Supported tasks     | retrieval_document, retrieval_query, semantic_similarity, classification, clustering |
| Matryoshka support  | Yes (truncate to 768, 1536, or 3072)  |
| API name            | models/gemini-embedding-exp-03-07     |
| Rate limit (free)   | 1,500 requests/minute                 |

**Performance**: Ranks #1 on the MTEB (Massive Text Embedding Benchmark) leaderboard as of
March 2025, achieving a mean score of 72.3 across 56 tasks.

### text-embedding-004

The previous generation production embedding model.

| Property            | Value                                 |
|---------------------|---------------------------------------|
| Output dimension    | 768                                   |
| Max input tokens    | 2048                                  |
| API name            | models/text-embedding-004             |

---

## Task Types

Using the correct `task_type` parameter is critical for retrieval quality:

- **retrieval_document**: Use when embedding documents/chunks for storage in the vector index.
- **retrieval_query**: Use when embedding the user's search query at inference time.
- **semantic_similarity**: Use for measuring similarity between two texts.
- **classification**: Use for text classification tasks.
- **clustering**: Use when clustering documents by topic.

Using mismatched task types (e.g., embedding queries with `retrieval_document`) degrades
retrieval quality by 10-25% on standard benchmarks.

---

## Python Integration

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GOOGLE_API_KEY")

# Embed a document for storage
doc_result = genai.embed_content(
    model="models/gemini-embedding-exp-03-07",
    content="Retrieval-augmented generation combines search with language models.",
    task_type="retrieval_document",
)
doc_vector = doc_result["embedding"]   # list of 3072 floats

# Embed a query at search time
query_result = genai.embed_content(
    model="models/gemini-embedding-exp-03-07",
    content="How does RAG work?",
    task_type="retrieval_query",
)
query_vector = query_result["embedding"]   # list of 3072 floats
```

---

## Matryoshka Embeddings

gemini-embedding-exp-03-07 supports Matryoshka Representation Learning (MRL), meaning the
first N dimensions of the 3072-dim vector form a valid lower-dimensional embedding. This
allows you to trade retrieval quality for storage/compute savings:

| Truncated Dimensions | Relative Quality | Storage (per vector) |
|----------------------|-----------------|----------------------|
| 3072                 | 100%            | 12 KB (float32)      |
| 1536                 | ~97%            | 6 KB                 |
| 768                  | ~94%            | 3 KB                 |

For most RAG pipelines, 768 dimensions is a practical default that halves storage while
retaining 94% of retrieval quality.

---

## ChromaDB Integration

```python
import chromadb
import google.generativeai as genai

genai.configure(api_key="YOUR_GOOGLE_API_KEY")

class GeminiEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        results = []
        for text in input:
            r = genai.embed_content(
                model="models/gemini-embedding-exp-03-07",
                content=text,
                task_type="retrieval_document",
            )
            results.append(r["embedding"])
        return results

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=GeminiEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)
```

---

## Rate Limits and Batching

The free tier allows 1,500 embedding requests per minute. For large ingestion jobs:

1. Process documents in batches of 32 (recommended for 3072-dim model)
2. Add exponential backoff on `ResourceExhausted` errors
3. Cache embeddings locally to avoid recomputing on restarts

---

## Cost Comparison

| Model                          | Price per 1M tokens |
|--------------------------------|---------------------|
| gemini-embedding-exp-03-07     | Free (preview)      |
| text-embedding-004             | $0.00025            |
| OpenAI text-embedding-3-large  | $0.13               |
| Cohere embed-v3                | $0.10               |

---

## Best Practices

1. Always use `retrieval_query` task type for query embeddings and `retrieval_document` for chunks.
2. Pre-process text before embedding: remove boilerplate, normalize whitespace.
3. Chunk size should be ≤ 512 tokens for best embedding quality (model trained on shorter passages).
4. Store embeddings in float32 (not float16) to preserve cosine similarity precision.
5. Rebuild the index if you switch embedding models — old and new vectors are not compatible.
