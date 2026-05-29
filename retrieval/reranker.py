from typing import Optional
import cohere

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CohereReranker:
    """
    Re-ranks retrieved chunks using Cohere's reranking API.
    Improves precision by re-scoring with a cross-encoder.
    """

    def __init__(
        self,
        api_key: str = settings.COHERE_API_KEY,
        model: str = "rerank-english-v3.0",
        top_n: int = None,
    ):
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n or settings.TOP_K

    def rerank(self, query: str, chunks: list[dict], top_n: int = None) -> list[dict]:
        """
        Rerank chunks by relevance to query.
        Returns re-ranked list of chunk dicts with 'rerank_score' added.
        """
        if not chunks:
            return []

        n = top_n or self.top_n
        documents = [c["content"] for c in chunks]

        logger.info(f"Reranking {len(chunks)} chunks with Cohere (top_n={n})")

        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=n,
            )
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}. Returning original order.")
            return chunks[:n]

        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        return reranked


class SimpleReranker:
    """
    Fallback reranker using keyword overlap scoring (no external API needed).
    """

    def rerank(self, query: str, chunks: list[dict], top_n: int = settings.TOP_K) -> list[dict]:
        query_tokens = set(query.lower().split())
        scored = []
        for chunk in chunks:
            content_tokens = set(chunk["content"].lower().split())
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            scored_chunk = chunk.copy()
            scored_chunk["rerank_score"] = overlap
            scored.append(scored_chunk)

        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_n]
