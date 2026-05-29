from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings
from utils.helpers import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Model RAG API powered by Gemini, Groq, and Anthropic",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    # Ensure required directories exist
    for d in [
        settings.DATA_DIR,
        settings.DOCUMENTS_DIR,
        settings.IMAGES_DIR,
        settings.VIDEOS_DIR,
        settings.CHROMA_PERSIST_DIR,
    ]:
        ensure_dir(d)
    logger.info("Directories initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
