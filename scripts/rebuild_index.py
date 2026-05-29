#!/usr/bin/env python3
"""
CLI script to wipe and rebuild the entire ChromaDB vector index.

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --dir ./data/documents --confirm
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_pipeline import RAGPipeline
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("rebuild_index")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the ChromaDB vector index from scratch"
    )
    parser.add_argument(
        "--dir",
        default=settings.DOCUMENTS_DIR,
        help=f"Documents directory to re-ingest (default: {settings.DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    doc_dir = os.path.abspath(args.dir)

    if not os.path.isdir(doc_dir):
        logger.error(f"Directory not found: {doc_dir}")
        sys.exit(1)

    if not args.confirm:
        print(
            f"\n⚠️  WARNING: This will DELETE the entire ChromaDB collection "
            f"('{settings.CHROMA_COLLECTION_NAME}') and re-ingest all documents "
            f"from:\n  {doc_dir}\n"
        )
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    logger.warning(f"Deleting ChromaDB collection: {settings.CHROMA_COLLECTION_NAME}")
    pipeline.vectorstore.delete_collection()

    # Reinitialize collection after deletion
    from vectorstore.chroma_client import ChromaVectorStore
    pipeline.vectorstore = ChromaVectorStore()
    pipeline.dense_retriever.vectorstore = pipeline.vectorstore
    pipeline.retriever.vectorstore = pipeline.vectorstore

    logger.info(f"Re-ingesting documents from: {doc_dir}")
    n = pipeline.ingest_directory(doc_dir)

    stats = pipeline.collection_stats()
    logger.info(
        f"Index rebuild complete | "
        f"Chunks stored: {n} | "
        f"Sources: {len(stats['sources'])}"
    )
    print(f"\n✓ Rebuild complete. {n} chunks stored from {len(stats['sources'])} sources.")


if __name__ == "__main__":
    main()
