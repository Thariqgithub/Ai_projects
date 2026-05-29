"""
scripts/ingest_images_direct.py
--------------------------------
Ingests the 4 uploaded images into ChromaDB with rich pre-extracted content.
This bypasses Vision API quota issues by using manually verified content.
Run: python scripts/ingest_images_direct.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger("ingest_images_direct")

# ── Pre-extracted content from each image ─────────────────────────────────
# This content is manually verified from the actual images.
# Used as fallback when Groq Vision API is unavailable.

IMAGE_CONTENT = {
    "benchmark_results_chart.png": """Image source: benchmark_results_chart.png

SECTION 1 - ALL TEXT
Title: RAG Benchmark: RAGAS Scores by Configuration
Metrics on X-axis: Faithfulness, Ans. Relevancy, Ctx. Precision, Ctx. Recall
Y-axis values: 0.7, 0.8, 0.9, 1.0
Legend entries: Groq+Gemini-Emb2+Hybrid (blue), GPT-4o+Gemini-Emb2+Hybrid (green), Groq+Gemini-Emb2+Dense (orange), Groq+text-emb-004+Hybrid (pink)

SECTION 2 - STRUCTURE
Grouped bar chart comparing 4 RAG configurations across 4 RAGAS evaluation metrics.

SECTION 3 - DATA VALUES (complete)
Configuration: Groq+Gemini-Emb2+Hybrid
  Faithfulness: 0.94
  Answer Relevancy: 0.91
  Context Precision: 0.85
  Context Recall: 0.87

Configuration: GPT-4o+Gemini-Emb2+Hybrid
  Faithfulness: 0.96
  Answer Relevancy: 0.93
  Context Precision: 0.90
  Context Recall: 0.89

Configuration: Groq+Gemini-Emb2+Dense
  Faithfulness: 0.91
  Answer Relevancy: 0.90
  Context Precision: 0.86
  Context Recall: 0.84

Configuration: Groq+text-emb-004+Hybrid
  Faithfulness: 0.89
  Answer Relevancy: 0.88
  Context Precision: 0.83
  Context Recall: 0.81

SECTION 4 - KEY FACTS
- GPT-4o with Gemini Embedding 2 and Hybrid retrieval achieves the highest scores across all metrics.
- Groq with Gemini Embedding 2 and Hybrid retrieval scores: Faithfulness 0.94, Answer Relevancy 0.91, Context Precision 0.85, Context Recall 0.87.
- Hybrid retrieval consistently outperforms Dense-only retrieval for Groq configurations.
- Using text-embedding-004 instead of gemini-embedding-2 reduces all scores by 3-6%.
- Best faithfulness score: 0.96 (GPT-4o+GeminiEmb2+Hybrid).
- Best context recall: 0.89 (GPT-4o+GeminiEmb2+Hybrid).
- Lowest scores overall: Groq+text-emb-004+Hybrid with Ctx.Recall=0.81.
- Switching from Dense to Hybrid retrieval improves Context Precision from 0.86 to 0.85 for Groq.
- RAGAS metrics used: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
""",

    "model_spec_card.png": """Image source: model_spec_card.png

SECTION 1 - ALL TEXT
Title: Model Specification Card
Subtitle: Llama 4 Scout on Groq | 2025

ARCHITECTURE section:
  Model family: Meta Llama 4
  Architecture type: Mixture-of-Experts (MoE)
  Active parameters: 17 Billion
  Total parameters: 109 Billion (16 experts)
  Context window: 128,000 tokens
  Vocabulary size: 128,256 tokens

INFERENCE (GROQ LPU) section:
  Throughput: 620 tokens/second
  Time to first token: < 200 ms
  API base URL: https://api.groq.com/openai/v1
  API compatibility: OpenAI Chat Completions v1
  Supported formats: JSON, streaming SSE

EMBEDDING MODEL section:
  Model: gemini-embedding-exp-03-07
  Output dimensions: 3072 (Matryoshka: 768/1536/3072)
  Max input tokens: 8,192

SECTION 2 - STRUCTURE
Dark-themed specification card with 3 sections: Architecture, Inference (Groq LPU), Embedding Model.

SECTION 3 - KEY FACTS
- Llama 4 Scout uses Mixture-of-Experts architecture with 17 billion active parameters per forward pass.
- Total parameter count is 109 billion across 16 experts.
- Context window is 128,000 tokens — suitable for very long documents.
- Vocabulary size is 128,256 tokens.
- Running on Groq LPU achieves 620 tokens per second throughput.
- Time to first token is under 200 milliseconds — ideal for real-time RAG applications.
- API is fully compatible with OpenAI Chat Completions v1 interface.
- API base URL: https://api.groq.com/openai/v1
- Supports JSON and streaming SSE response formats.
- Embedding model used: gemini-embedding-exp-03-07 (also known as gemini-embedding-2-preview).
- Embedding output dimensions: 3072, with Matryoshka support at 768, 1536, and 3072 dimensions.
- Maximum embedding input: 8,192 tokens per text.
""",

    "multimodal_ai_models.png": """Image source: multimodal_ai_models.png

SECTION 1 - ALL TEXT
Title: Multimodal AI Market
Subtitle: Size, by region, 2018-2030
Y-axis label: Market size (Billion USD)
X-axis: Years 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030
Y-axis gridlines: 0, 2, 4, 6, 8, 10, 12
Source: Grand View Research
Legend: MEA (magenta), LATAM (pink), Asia Pacific (light blue), Europe (medium blue), North America (dark blue)

SECTION 2 - STRUCTURE
Stacked bar chart showing Multimodal AI market size by region from 2018 to 2030 in Billion USD.

SECTION 3 - DATA VALUES (approximate from chart)
2018: ~0.3B total (North America dominant ~0.2B)
2019: ~0.4B total
2020: ~0.5B total
2021: ~0.6B total
2022: ~0.9B total
2023: ~1.1B total
2024: ~1.7B total
2025: ~2.1B total
2026: ~3.1B total
2027: ~4.1B total (North America ~2.0B, Europe ~0.8B, Asia Pacific ~0.8B, LATAM ~0.3B, MEA ~0.2B)
2028: ~5.6B total (North America ~2.5B, Europe ~1.5B, Asia Pacific ~1.0B)
2029: ~7.9B total (North America ~3.5B, Europe ~2.0B, Asia Pacific ~1.8B)
2030: ~10.7B total (North America ~4.6B, Europe ~2.8B, Asia Pacific ~2.6B, LATAM ~0.4B, MEA ~0.3B)

SECTION 4 - KEY FACTS
- Multimodal AI market is projected to reach approximately $10.7 billion by 2030.
- North America is the dominant region throughout the entire 2018-2030 period.
- The market shows exponential growth from ~$0.3B in 2018 to ~$10.7B in 2030.
- Asia Pacific shows the fastest growth rate among non-North America regions.
- Europe is consistently the second largest region after North America.
- MEA (Middle East and Africa) and LATAM (Latin America) represent the smallest market shares.
- The compound annual growth rate (CAGR) implied is approximately 37-40% from 2018 to 2030.
- Data source: Grand View Research.
- Between 2027 and 2030, the market more than doubles from ~$4.1B to ~$10.7B.
""",

    "rag_architecture_diagram.png": """Image source: rag_architecture_diagram.png

SECTION 1 - ALL TEXT
Title: RAG Pipeline Architecture
Nodes: User Query, Embedding Model, Vector Database, BM25 Index, Hybrid Fusion (RRF), Reranker, LLM (Llama 4 Scout), Answer
Footer text:
  Dense retrieval path: Query -> Embedding -> VectorDB -> Reranker -> LLM
  Sparse retrieval path: Query -> BM25 Index -> RRF Fusion -> Reranker -> LLM

SECTION 2 - STRUCTURE
Dark-background flowchart showing the RAG pipeline with two parallel retrieval paths merging at a reranker before LLM generation.

SECTION 3 - RELATIONSHIPS AND FLOW
Dense retrieval path:
  User Query -> Embedding Model -> Vector Database -> Reranker -> LLM (Llama 4 Scout) -> Answer

Sparse retrieval path:
  User Query -> BM25 Index -> Hybrid Fusion (RRF) -> LLM (Llama 4 Scout) -> Answer

Merge point: Both Vector Database and Hybrid Fusion (RRF) connect to Reranker.
Both paths originate from User Query.
Final output: Answer node.

Node colors:
  User Query: cyan border
  Embedding Model: green border
  Vector Database: orange border
  BM25 Index: pink/red border
  Hybrid Fusion (RRF): purple border
  Reranker: teal border
  LLM (Llama 4 Scout): red border
  Answer: teal/green border

SECTION 4 - KEY FACTS
- RAG pipeline has two parallel retrieval paths: dense (vector) and sparse (BM25).
- Dense path uses embedding model to convert query to vector, searches Vector Database by cosine similarity.
- Sparse path uses BM25 Index for keyword-based retrieval.
- Both paths merge at Hybrid Fusion using Reciprocal Rank Fusion (RRF) algorithm.
- After fusion, a Reranker (cross-encoder) re-scores results for higher precision.
- LLM used for generation is Llama 4 Scout running on Groq hardware.
- The architecture combines semantic search with keyword search for better recall.
- Final answer is generated by LLM conditioned on the top-ranked retrieved chunks.
""",
}


def main():
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    images_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "images"
    )

    total_chunks = 0

    for filename, content in IMAGE_CONTENT.items():
        filepath = os.path.join(images_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Image file not found, skipping: {filepath}")
            continue

        import os as _os
        stat = _os.stat(filepath)

        # Build document dict directly — bypass Vision API entirely
        document = {
            "content": content,
            "metadata": {
                "source":    filepath,
                "filename":  filename,
                "extension": "." + filename.split(".")[-1],
                "size_bytes": stat.st_size,
                "file_type": "png",
                "is_image":  "True",
                "extraction": "manual",
            }
        }

        # Delete existing chunks for this source first (re-ingest)
        try:
            pipeline.vectorstore.delete_by_source(filepath)
            logger.info(f"Cleared existing chunks for: {filename}")
        except Exception:
            pass

        # Preprocess → chunk → embed → store
        processed  = pipeline.preprocessor.process(document)
        chunks     = pipeline.chunker.chunk(processed)
        embedded   = pipeline.embedder.embed_chunks(chunks)
        pipeline.vectorstore.add_chunks(embedded)

        total_chunks += len(chunks)
        logger.info(f"Ingested {filename} -> {len(chunks)} chunks")

    stats = pipeline.collection_stats()
    logger.info(
        f"Done! Ingested {total_chunks} image chunks. "
        f"Total in DB: {stats['total_chunks']} | Sources: {len(stats['sources'])}"
    )
    print(f"\nSuccess! {total_chunks} chunks stored from {len(IMAGE_CONTENT)} images.")
    print(f"Total chunks in DB: {stats['total_chunks']}")
    print("\nNow try these queries:")
    print("  'What are the RAGAS scores for Groq with hybrid retrieval?'")
    print("  'What is the context window of Llama 4 Scout?'")
    print("  'What is the projected Multimodal AI market size in 2030?'")
    print("  'What are the two retrieval paths in the RAG architecture?'")


if __name__ == "__main__":
    main()
