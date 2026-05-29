import base64
import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from groq import Groq

from api.schemas import (
    IngestTextRequest, IngestResponse,
    QueryRequest, QueryResponse, SourceRef,
    CollectionStatsResponse, DeleteSourceRequest, DeleteResponse, HealthResponse,
)
from rag.rag_pipeline import RAGPipeline
from config.settings import settings
from utils.helpers import ensure_dir, sanitize_filename
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_pipeline: RAGPipeline = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _save_upload(file: UploadFile, directory: str) -> str:
    """
    Save an uploaded file to directory and return the ABSOLUTE path.
    Uses os.path.abspath to avoid any relative-path lookup failures.
    """
    ensure_dir(directory)
    safe_name = sanitize_filename(file.filename or "upload")
    dest = os.path.abspath(os.path.join(directory, safe_name))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if not os.path.exists(dest):
        raise FileNotFoundError(f"File was not saved correctly: {dest}")
    logger.info(f"Saved upload to: {dest} ({os.path.getsize(dest)} bytes)")
    return dest


# ── Health ──────────────────────────────────────────────────────────────── #

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    stats = get_pipeline().collection_stats()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        total_chunks=stats["total_chunks"],
    )


# ── Ingest ──────────────────────────────────────────────────────────────── #

@router.post("/ingest/image", response_model=IngestResponse, tags=["Ingest"])
async def ingest_image(file: UploadFile = File(...)):
    """
    Upload an image (PNG/JPG/WEBP):
    1. Save to data/images/
    2. Analyze with Groq Vision (llama-4-scout) — free, no quota issues
    3. Embed extracted text with gemini-embedding-2-preview
    4. Store chunks in ChromaDB
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Use PNG, JPG, or WEBP."
        )

    try:
        dest = _save_upload(file, settings.IMAGES_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    try:
        n = get_pipeline().ingest_file(dest)
    except Exception as e:
        logger.error(f"Image ingest failed for {dest}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        status="success",
        chunks_stored=n,
        message=f"Image '{os.path.basename(dest)}' analyzed with Groq Vision -> {n} chunks stored",
    )


@router.post("/ingest/file", response_model=IngestResponse, tags=["Ingest"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest any document (PDF, TXT, DOCX, MD, CSV).
    Images sent here are automatically routed to data/images/ and
    processed with Groq Vision — no need to use /ingest/image separately.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    is_image = ext in {".png", ".jpg", ".jpeg", ".webp"}

    # Route images to images dir, documents to documents dir
    save_dir = settings.IMAGES_DIR if is_image else settings.DOCUMENTS_DIR

    try:
        dest = _save_upload(file, save_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        n = get_pipeline().ingest_file(dest)
    except Exception as e:
        logger.error(f"Ingest failed for {dest}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    action = "analyzed with Groq Vision" if is_image else "ingested"
    return IngestResponse(
        status="success",
        chunks_stored=n,
        message=f"'{os.path.basename(dest)}' {action} -> {n} chunks stored",
    )


@router.post("/analyze/image", tags=["Analyze"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image with Groq Vision and return description.
    Does NOT store in ChromaDB — for one-off image inspection.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")
    image_bytes = await file.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analyze this image thoroughly. Describe everything you see: "
                        "all text, numbers, data, diagrams, charts, tables, "
                        "relationships, and key facts. Be detailed and specific."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{b64}"
                    }},
                ],
            }],
            max_tokens=2048,
            temperature=0.1,
        )
        return {
            "filename": file.filename,
            "analysis": response.choices[0].message.content,
            "model": settings.GROQ_MODEL,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/text", response_model=IngestResponse, tags=["Ingest"])
def ingest_text(request: IngestTextRequest):
    """Ingest raw text directly."""
    try:
        n = get_pipeline().ingest_text(request.text, request.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(
        status="success",
        chunks_stored=n,
        message=f"Ingested inline text -> {n} chunks",
    )


@router.post("/ingest/directory", response_model=IngestResponse, tags=["Ingest"])
def ingest_directory(directory: str = settings.DOCUMENTS_DIR):
    """Ingest all documents from a directory path."""
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    try:
        n = get_pipeline().ingest_directory(directory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(
        status="success",
        chunks_stored=n,
        message=f"Ingested directory '{directory}' -> {n} chunks",
    )


# ── Query ────────────────────────────────────────────────────────────────── #

@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """RAG query: retrieve + rerank + generate."""
    try:
        result = get_pipeline().query(
            question=request.question,
            top_k=request.top_k,
            llm_model=request.llm_model,
            filters=request.filters,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    sources = [SourceRef(**s) for s in result["sources"]]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        question=request.question,
        model_used=request.llm_model or settings.DEFAULT_LLM,
    )


@router.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """Streaming RAG query (Server-Sent Events)."""
    async def event_generator():
        try:
            async for token in get_pipeline().stream_query(
                question=request.question,
                top_k=request.top_k,
                llm_model=request.llm_model,
                filters=request.filters,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Collection ───────────────────────────────────────────────────────────── #

@router.get("/collection/stats", response_model=CollectionStatsResponse, tags=["Collection"])
def collection_stats():
    stats = get_pipeline().collection_stats()
    return CollectionStatsResponse(**stats)


@router.delete("/collection/source", response_model=DeleteResponse, tags=["Collection"])
def delete_source(request: DeleteSourceRequest):
    """Delete all chunks from a specific source file."""
    try:
        get_pipeline().delete_source(request.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return DeleteResponse(
        status="success",
        message=f"Deleted all chunks for source: {request.source}",
    )