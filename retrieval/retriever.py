from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.chroma_client import ChromaVectorStore
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Performs dense vector retrieval using Gemini embeddings + ChromaDB.
    """

    def __init__(
        self,
        embedder: GeminiEmbeddings = None,
        vectorstore: ChromaVectorStore = None,
        top_k: int = settings.TOP_K,
    ):
        self.embedder = embedder or GeminiEmbeddings()
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None, filters: dict = None) -> list[dict]:
        """
        Retrieve top-k chunks relevant to the query.
        Returns list of dicts with id, content, metadata, score.
        """
        k = top_k or self.top_k
        logger.info(f"Dense retrieval: query='{query[:60]}...', top_k={k}")

        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=k,
            where=filters,
        )

        logger.debug(f"Retrieved {len(results)} chunks")
        return results
