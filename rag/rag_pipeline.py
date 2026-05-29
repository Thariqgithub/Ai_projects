from typing import Optional, AsyncGenerator

from ingestion.loader import DocumentLoader
from ingestion.preprocess import TextPreprocessor
from ingestion.chunker import TextChunker
from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.chroma_client import ChromaVectorStore
from retrieval.hybrid_search import HybridSearchRetriever
from retrieval.reranker import CohereReranker, SimpleReranker
from retrieval.retriever import DenseRetriever
from router.llm_router import LLMRouter
from rag.prompt_builder import PromptBuilder
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
      ingest → embed → store → retrieve → rerank → prompt → generate
    """

    def __init__(
        self,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        llm_model: Optional[str] = None,
    ):
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.embedder = GeminiEmbeddings()
        self.vectorstore = ChromaVectorStore()
        self.dense_retriever = DenseRetriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        self.retriever = (
            HybridSearchRetriever(
                dense_retriever=self.dense_retriever,
                vectorstore=self.vectorstore,
            )
            if use_hybrid
            else self.dense_retriever
        )

        try:
            self.reranker = CohereReranker() if use_reranker else None
        except Exception:
            logger.warning("Cohere reranker unavailable, using SimpleReranker.")
            self.reranker = SimpleReranker() if use_reranker else None

        self.router = LLMRouter(default_model=llm_model or settings.DEFAULT_LLM)
        self.prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------ #
    # INGESTION                                                            #
    # ------------------------------------------------------------------ #

    def ingest_file(self, filepath: str) -> int:
        """Ingest a single file. Returns number of chunks stored."""
        logger.info(f"Ingesting file: {filepath}")
        raw = self.loader.load(filepath)
        processed = self.preprocessor.process(raw)
        chunks = self.chunker.chunk(processed)
        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.vectorstore.add_chunks(embedded_chunks)
        logger.info(f"Ingested {len(chunks)} chunks from {filepath}")
        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """Ingest all supported files in a directory. Returns total chunks."""
        logger.info(f"Ingesting directory: {directory}")
        raw_docs = self.loader.load_directory(directory)
        processed_docs = self.preprocessor.process_many(raw_docs)
        all_chunks = self.chunker.chunk_many(processed_docs)
        embedded_chunks = self.embedder.embed_chunks(all_chunks)
        self.vectorstore.add_chunks(embedded_chunks)
        logger.info(f"Ingested {len(all_chunks)} total chunks from {directory}")
        return len(all_chunks)

    def ingest_text(self, text: str, metadata: dict = None) -> int:
        """Ingest raw text directly."""
        document = {"content": text, "metadata": metadata or {"source": "inline"}}
        processed = self.preprocessor.process(document)
        chunks = self.chunker.chunk(processed)
        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.vectorstore.add_chunks(embedded_chunks)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # QUERY                                                                #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        llm_model: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Full RAG query pipeline.
        Returns: {"answer": str, "sources": list[dict], "chunks": list[dict]}
        """
        logger.info(f"RAG query: '{question[:80]}...'")

        # Retrieval
        chunks = self.retriever.retrieve(question, top_k=top_k * 2, filters=filters)

        # Reranking
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_n=top_k)
        else:
            chunks = chunks[:top_k]

        if not chunks:
            return {
                "answer": "No relevant context found in the knowledge base.",
                "sources": [],
                "chunks": [],
            }

        # Prompt building
        prompt = self.prompt_builder.build(question, chunks)

        # LLM generation
        answer = self.router.generate(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            model=llm_model,
        )

        sources = [
            {
                "id": c["id"],
                "source": c["metadata"].get("source", ""),
                "filename": c["metadata"].get("filename", ""),
                "score": c.get("rerank_score") or c.get("hybrid_score") or c.get("score", 0),
            }
            for c in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
        }

    async def stream_query(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        llm_model: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG query."""
        logger.info(f"Streaming RAG query: '{question[:80]}...'")

        chunks = self.retriever.retrieve(question, top_k=top_k * 2, filters=filters)
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_n=top_k)
        else:
            chunks = chunks[:top_k]

        if not chunks:
            yield "No relevant context found in the knowledge base."
            return

        prompt = self.prompt_builder.build(question, chunks)

        async for token in self.router.stream(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            model=llm_model,
        ):
            yield token

    # ------------------------------------------------------------------ #
    # UTILS                                                                #
    # ------------------------------------------------------------------ #

    def delete_source(self, source: str) -> None:
        self.vectorstore.delete_by_source(source)

    def collection_stats(self) -> dict:
        return {
            "total_chunks": self.vectorstore.count(),
            "sources": self.vectorstore.list_sources(),
        }
