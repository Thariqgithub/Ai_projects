import re
import time

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError(
        "Could not import 'google.genai'. Run fix_env.bat or:\n"
        "  pip uninstall google-generativeai -y && pip install 'google-genai>=1.5.0'"
    )

from config.settings import settings
from utils.helpers import chunk_list
from utils.logger import get_logger

logger = get_logger(__name__)

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# gemini-embedding-2-preview free tier: 100 requests/minute
# We use batch_size=10 and pace at 80 RPM → safe headroom
_BATCH_INTERVAL = 60.0 / 80   # 0.75s between batches
_MAX_RETRIES    = 8            # enough for large docs like 861-chunk PDFs


class GeminiEmbeddings:
    """
    Generates embeddings using Google's gemini-embedding-2-preview model.

    Rate limit handling:
      - Paces requests at 80 RPM (free tier limit is 100 RPM)
      - On 429, parses the retryDelay from the error and waits exactly that long
      - Retries up to 8 times — enough for any size document
    """

    _TASK_MAP = {
        "retrieval_document":  "RETRIEVAL_DOCUMENT",
        "retrieval_query":     "RETRIEVAL_QUERY",
        "semantic_similarity": "SEMANTIC_SIMILARITY",
        "classification":      "CLASSIFICATION",
        "clustering":          "CLUSTERING",
    }

    def __init__(
        self,
        model: str = settings.EMBEDDING_MODEL,
        task_type: str = "retrieval_document",
        batch_size: int = 10,   # 10 texts per request — stays well under rate limits
    ):
        self.model     = model
        self.task_type = task_type
        self.batch_size = batch_size
        self._last_request_time = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def embed_text(self, text: str, task_type: str = None) -> list[float]:
        """Embed a single text string."""
        return self._embed_with_retry([text], task_type or self.task_type)[0]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        return self._embed_with_retry([query], "retrieval_query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in rate-limited batches."""
        all_embeddings: list[list[float]] = []
        batches = chunk_list(texts, self.batch_size)
        total   = len(batches)

        logger.info(
            f"Embedding {len(texts)} texts in {total} batches "
            f"(model={self.model}, batch_size={self.batch_size})"
        )

        for i, batch in enumerate(batches):
            logger.info(f"Embedding batch {i+1}/{total} ({len(batch)} texts)")
            embeddings = self._embed_with_retry(batch, self.task_type)
            all_embeddings.extend(embeddings)
            # Pace requests to stay under rate limit
            self._rate_limit_wait()

        return all_embeddings

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Add 'embedding' field to each chunk dict in-place."""
        texts      = [c["content"] for c in chunks]
        embeddings = self.embed_documents(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
        logger.info(
            f"Generated embeddings for {len(chunks)} chunks "
            f"(dim={len(embeddings[0]) if embeddings else 0})"
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _rate_limit_wait(self):
        """Ensure minimum interval between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _BATCH_INTERVAL:
            time.sleep(_BATCH_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _embed_with_retry(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """
        Embed texts with automatic 429 retry using the server's retryDelay.
        """
        task = self._TASK_MAP.get(task_type, "RETRIEVAL_DOCUMENT")

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = _client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(task_type=task),
                )
                self._last_request_time = time.time()
                return [e.values for e in response.embeddings]

            except Exception as e:
                err_str = str(e)

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Parse retryDelay from error message
                    delay = self._parse_retry_delay(err_str)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{_MAX_RETRIES}). "
                        f"Waiting {delay}s as instructed by API..."
                    )
                    time.sleep(delay)
                    continue

                # Non-rate-limit error — fail immediately
                logger.error(f"Embedding error (attempt {attempt}): {e}")
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(2.0)

        raise RuntimeError(
            f"Embedding failed after {_MAX_RETRIES} attempts. "
            f"Consider enabling billing on Google AI Studio for higher limits."
        )

    @staticmethod
    def _parse_retry_delay(err_str: str) -> float:
        """
        Extract retryDelay from the API error message.
        Falls back to 60s if not found.
        """
        # Try to find retryDelay value like: 'retryDelay': '39s'
        match = re.search(r"retryDelay.*?['\"](\d+(?:\.\d+)?)s['\"]", err_str)
        if match:
            return float(match.group(1)) + 2  # add 2s buffer
        # Also try plain number pattern: "Please retry in 39.6s"
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
        if match:
            return float(match.group(1)) + 2
        return 62.0  # safe default: wait just over 1 minute