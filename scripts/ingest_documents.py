#!/usr/bin/env python3
"""
CLI script to ingest documents into the RAG pipeline.

Usage:
    python scripts/ingest_documents.py --path ./data/documents
    python scripts/ingest_documents.py --path ./data/documents/report.pdf
    python scripts/ingest_documents.py --path ./data/documents --chunk-size 512 --overlap 64
"""

import argparse
import io
import os
import sys
import time

# Force UTF-8 output on Windows (fixes cp1252 UnicodeEncodeError in PowerShell/CMD)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger("ingest_documents")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Multi-Model RAG pipeline"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a file or directory to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Chunk overlap in characters (default: 64)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "sentence", "fixed"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "anthropic"],
        default=None,
        help="LLM to use for generation (default: auto-route)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without storing embeddings",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = os.path.abspath(args.path)

    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(llm_model=args.llm)

    # Override chunker settings
    pipeline.chunker.chunk_size = args.chunk_size
    pipeline.chunker.chunk_overlap = args.overlap
    pipeline.chunker.strategy = args.strategy

    start = time.time()

    if args.dry_run:
        logger.info("[DRY RUN] Loading and chunking only — no embeddings stored")
        if os.path.isfile(path):
            raw = pipeline.loader.load(path)
            processed = pipeline.preprocessor.process(raw)
            chunks = pipeline.chunker.chunk(processed)
            logger.info(f"[DRY RUN] Would store {len(chunks)} chunks from {path}")
        else:
            raw_docs = pipeline.loader.load_directory(path)
            processed_docs = pipeline.preprocessor.process_many(raw_docs)
            all_chunks = pipeline.chunker.chunk_many(processed_docs)
            logger.info(f"[DRY RUN] Would store {len(all_chunks)} chunks from {path}")
        return

    if os.path.isfile(path):
        logger.info(f"Ingesting file: {path}")
        n = pipeline.ingest_file(path)
        logger.info(f"✓ Stored {n} chunks from file")
    elif os.path.isdir(path):
        logger.info(f"Ingesting directory: {path}")
        n = pipeline.ingest_directory(path)
        logger.info(f"✓ Stored {n} total chunks from directory")
    else:
        logger.error(f"Not a valid file or directory: {path}")
        sys.exit(1)

    elapsed = time.time() - start
    stats = pipeline.collection_stats()
    logger.info(
        f"Done in {elapsed:.1f}s | "
        f"Total chunks in DB: {stats['total_chunks']} | "
        f"Sources: {len(stats['sources'])}"
    )


if __name__ == "__main__":
    main()
