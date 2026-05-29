from workers.celery_worker import celery_app
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_pipeline():
    # Import lazily to avoid circular dependency at module load
    from rag.rag_pipeline import RAGPipeline
    return RAGPipeline()


@celery_app.task(bind=True, name="tasks.ingest_file")
def ingest_file_task(self, filepath: str) -> dict:
    """Background task to ingest a single file."""
    logger.info(f"[Task {self.request.id}] Ingesting file: {filepath}")
    try:
        pipeline = _get_pipeline()
        n = pipeline.ingest_file(filepath)
        return {"status": "success", "filepath": filepath, "chunks": n}
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Ingest failed: {e}")
        raise self.retry(exc=e, countdown=5, max_retries=3)


@celery_app.task(bind=True, name="tasks.ingest_directory")
def ingest_directory_task(self, directory: str) -> dict:
    """Background task to ingest a directory of files."""
    logger.info(f"[Task {self.request.id}] Ingesting directory: {directory}")
    try:
        pipeline = _get_pipeline()
        n = pipeline.ingest_directory(directory)
        return {"status": "success", "directory": directory, "chunks": n}
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Directory ingest failed: {e}")
        raise self.retry(exc=e, countdown=10, max_retries=2)


@celery_app.task(bind=True, name="tasks.rebuild_index")
def rebuild_index_task(self, directory: str) -> dict:
    """
    Background task to wipe the collection and re-ingest all documents.
    Use with caution — destructive operation.
    """
    logger.warning(f"[Task {self.request.id}] Rebuilding index from: {directory}")
    try:
        pipeline = _get_pipeline()
        pipeline.vectorstore.delete_collection()
        n = pipeline.ingest_directory(directory)
        return {"status": "success", "chunks": n}
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Rebuild failed: {e}")
        raise


@celery_app.task(bind=True, name="tasks.rag_query")
def rag_query_task(self, question: str, top_k: int = 5, llm_model: str = None) -> dict:
    """Background RAG query task for async processing."""
    logger.info(f"[Task {self.request.id}] RAG query: '{question[:60]}'")
    try:
        pipeline = _get_pipeline()
        result = pipeline.query(question=question, top_k=top_k, llm_model=llm_model)
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Query failed: {e}")
        raise self.retry(exc=e, countdown=3, max_retries=2)
