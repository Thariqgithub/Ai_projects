from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    COHERE_API_KEY: str = ""

    # FastAPI
    APP_NAME: str = "Multi-Model RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # Embeddings — gemini-embedding-2-preview (Gemini Embedding 2)
    # First multimodal embedding model — text, image, video, audio, PDF
    # Output: 3072-dim by default; supports Matryoshka truncation to 768/1536
    EMBEDDING_MODEL: str = "gemini-embedding-2-preview"
    EMBEDDING_DIMENSION: int = 3072

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    TOP_K: int = 5
    HYBRID_ALPHA: float = 0.7  # Weight for dense vs sparse retrieval

    # LLM Router — Groq (llama-4-scout) is the primary LLM
    DEFAULT_LLM: str = "groq"  # groq | gemini
    GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GEMINI_MODEL: str = "gemini-2.0-flash"  # fallback generation model (new SDK name)

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # File storage
    DATA_DIR: str = "./data"
    DOCUMENTS_DIR: str = "./data/documents"
    IMAGES_DIR: str = "./data/images"
    VIDEOS_DIR: str = "./data/videos"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    # RAGAS Evaluation
    RAGAS_SAMPLE_SIZE: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
