from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Manages ChromaDB collection: add, query, delete, and list documents.
    """

    def __init__(
        self,
        persist_dir: str = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"persist_dir='{persist_dir}', "
            f"existing_docs={self.collection.count()}"
        )

    # ------------------------------------------------------------------ #
    # WRITE                                                                #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add embedded chunks to ChromaDB.
        Each chunk must have: id, content, embedding, metadata.
        """
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # Convert all metadata values to str/int/float (Chroma requirement)
        metadatas = [self._sanitize_metadata(m) for m in metadatas]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks associated with a specific source file."""
        results = self.collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for source: {source}")

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")

    # ------------------------------------------------------------------ #
    # READ                                                                 #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_embedding: list[float],
        top_k: int = settings.TOP_K,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Dense vector search.
        Returns a list of result dicts with id, content, metadata, distance.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def get_all(self) -> list[dict]:
        results = self.collection.get(include=["documents", "metadatas"])
        items = []
        for i, doc_id in enumerate(results["ids"]):
            items.append({
                "id": doc_id,
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return items

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results["metadatas"]}
        return sorted(sources)

    # ------------------------------------------------------------------ #
    # HELPERS                                                              #
    # ------------------------------------------------------------------ #

    def _format_results(self, raw: dict) -> list[dict]:
        formatted = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for doc_id, content, meta, dist in zip(ids, docs, metas, dists):
            formatted.append({
                "id": doc_id,
                "content": content,
                "metadata": meta,
                "score": 1 - dist,  # cosine similarity from distance
            })
        return formatted

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean
