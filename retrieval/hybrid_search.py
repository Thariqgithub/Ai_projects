from rank_bm25 import BM25Okapi

from retrieval.retriever import DenseRetriever
from vectorstore.chroma_client import ChromaVectorStore
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridSearchRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        vectorstore: ChromaVectorStore = None,
        alpha: float = settings.HYBRID_ALPHA,  # higher = more dense weight
        top_k: int = settings.TOP_K,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.vectorstore = vectorstore or self.dense.vectorstore
        self.alpha = alpha
        self.top_k = top_k
        self._bm25 = None
        self._all_chunks: list[dict] = []

    def _build_bm25_index(self) -> None:
        self._all_chunks = self.vectorstore.get_all()
        tokenized = [c["content"].lower().split() for c in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(self._all_chunks)} chunks")

    def retrieve(self, query: str, top_k: int = None, filters: dict = None) -> list[dict]:
        """
        Hybrid retrieval: dense + BM25, fused with RRF.
        """
        k = top_k or self.top_k

        # Dense retrieval (fetch more candidates)
        dense_results = self.dense.retrieve(query, top_k=k * 2, filters=filters)

        # Sparse BM25 retrieval
        if self._bm25 is None:
            self._build_bm25_index()

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        bm25_top_k_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:k * 2]

        sparse_results = [self._all_chunks[i] for i in bm25_top_k_idx]

        # Apply filters to sparse results
        if filters:
            sparse_results = [
                r for r in sparse_results
                if all(r["metadata"].get(fk) == fv for fk, fv in filters.items())
            ]

        # Reciprocal Rank Fusion
        fused = self._rrf(dense_results, sparse_results, k=60)
        return fused[:k]

    def _rrf(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.
        """
        scores: dict[str, float] = {}
        id_to_chunk: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0) + self.alpha * (1 / (k + rank + 1))
            id_to_chunk[cid] = chunk

        for rank, chunk in enumerate(sparse_results):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0) + (1 - self.alpha) * (1 / (k + rank + 1))
            id_to_chunk[cid] = chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        results = []
        for cid in sorted_ids:
            chunk = id_to_chunk[cid].copy()
            chunk["hybrid_score"] = scores[cid]
            results.append(chunk)

        return results
