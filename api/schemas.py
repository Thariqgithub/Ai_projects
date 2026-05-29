from pydantic import BaseModel, Field
from typing import Optional, List


# ── Ingest ────────────────────────────────────────────────────────────────── #

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text to ingest")
    metadata: Optional[dict] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────── #

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    llm_model: Optional[str] = Field(
        default=None,
        description="Override LLM: 'groq' | 'gemini'"
    )
    filters: Optional[dict] = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False)


class SourceRef(BaseModel):
    id: str
    source: str
    filename: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    question: str
    model_used: Optional[str] = None


# ── Collection ────────────────────────────────────────────────────────────── #

class CollectionStatsResponse(BaseModel):
    total_chunks: int
    sources: List[str]


class DeleteSourceRequest(BaseModel):
    source: str = Field(..., description="Source filepath to delete from index")


class DeleteResponse(BaseModel):
    status: str
    message: str


# ── Health ────────────────────────────────────────────────────────────────── #

class HealthResponse(BaseModel):
    status: str
    version: str
    total_chunks: int
