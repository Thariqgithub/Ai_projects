import os
import time

import fitz  # PyMuPDF
import docx
from PIL import Image

from utils.logger import get_logger
from utils.helpers import get_file_extension, file_size_mb
from config.settings import settings

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".png", ".jpg", ".jpeg", ".csv"}

# Correct model IDs for google-genai SDK (no "models/" prefix)
# Each has its own independent quota pool on the free tier
_VISION_MODELS = [
    "gemini-2.0-flash-lite",          # lightest — separate quota, very generous
    "gemini-1.5-flash-latest",        # correct name for new SDK
    "gemini-1.5-pro-latest",          # correct name for new SDK
    "gemini-2.0-flash",               # last resort (may be exhausted)
]

_VISION_PROMPT = """You are a content extraction engine for a RAG system.
Convert this image into the richest possible plain-text so it can be semantically searched.

SECTION 1 - ALL TEXT
List every word, number, label, title, axis value, legend entry, annotation, caption,
metric name, percentage, and score visible anywhere in the image.

SECTION 2 - STRUCTURE
Describe the visual type (flowchart, bar chart, table, architecture diagram, spec card, etc.)
and its main sections or components.

SECTION 3 - DATA VALUES
If a chart or table is present, transcribe ALL data row by row, column by column with exact values.
Example: "Groq Llama4 Scout | hybrid | chunk_size=512 | faithfulness=0.94 | answer_relevancy=0.91"

SECTION 4 - RELATIONSHIPS & FLOW
Describe every arrow, connection, or flow.
Example: "User Query -> Embedding Model -> Vector Database -> RRF Fusion -> Reranker -> LLM -> Answer"

SECTION 5 - KEY FACTS
List the most important facts, model names, scores, and configurations as direct searchable statements.

Be exhaustive. State facts directly without saying 'the image shows'."""


class DocumentLoader:
    """
    Loads various document types and returns raw text content.
    Images: Gemini Vision with automatic model fallback on quota/404 errors.
    """

    def __init__(self, use_vision: bool = True):
        self.use_vision = use_vision
        self._gemini_client = None

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._gemini_client

    def load(self, filepath: str) -> dict:
        ext = get_file_extension(filepath)
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        logger.info(f"Loading file: {filepath} ({file_size_mb(filepath):.2f} MB)")
        loaders = {
            ".pdf":  self._load_pdf,
            ".txt":  self._load_text,
            ".md":   self._load_text,
            ".docx": self._load_docx,
            ".csv":  self._load_text,
            ".png":  self._load_image,
            ".jpg":  self._load_image,
            ".jpeg": self._load_image,
        }
        content = loaders[ext](filepath)
        metadata = self._build_metadata(filepath, ext)
        return {"content": content, "metadata": metadata}

    def load_directory(self, directory: str) -> list[dict]:
        docs = []
        for root, _, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = get_file_extension(fpath)
                if ext in SUPPORTED_EXTENSIONS:
                    try:
                        docs.append(self.load(fpath))
                    except Exception as e:
                        logger.error(f"Failed to load {fpath}: {e}")
        logger.info(f"Loaded {len(docs)} documents from {directory}")
        return docs

    # ------------------------------------------------------------------ #
    # Loaders                                                              #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, filepath: str) -> str:
        text_parts = []
        with fitz.open(filepath) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    def _load_text(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_docx(self, filepath: str) -> str:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    def _load_image(self, filepath: str) -> str:
        """
        Try Gemini Vision models in order.
        - 404 NOT_FOUND  → skip immediately, try next model
        - 429 QUOTA      → wait retry delay, try next model
        - Other error    → log and try next model
        Falls back to Tesseract, then filename placeholder.
        """
        if self.use_vision and settings.GOOGLE_API_KEY:
            for model in _VISION_MODELS:
                result = self._try_vision_model(filepath, model)
                if result is not None:
                    return result
            logger.warning(
                f"All Vision models failed for {os.path.basename(filepath)}. "
                f"Trying Tesseract OCR."
            )
        return self._load_image_ocr(filepath)

    def _try_vision_model(self, filepath: str, model: str) -> str | None:
        """
        Returns extracted text on success.
        Returns None to signal caller to try next model.
        """
        from google.genai import types as genai_types

        with open(filepath, "rb") as f:
            image_bytes = f.read()

        ext = get_file_extension(filepath).lstrip(".")
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")

        try:
            logger.info(
                f"Trying Vision model '{model}' "
                f"for {os.path.basename(filepath)}"
            )
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=[
                    genai_types.Part.from_bytes(
                        data=image_bytes, mime_type=mime_type
                    ),
                    genai_types.Part.from_text(text=_VISION_PROMPT),
                ],
            )
            extracted = response.text.strip()
            filename = os.path.basename(filepath)
            full_content = f"Image source: {filename}\n\n{extracted}"
            logger.info(
                f"Vision '{model}' extracted "
                f"{len(full_content)} chars from {filename}"
            )
            return full_content

        except Exception as e:
            err_str = str(e)

            if "404" in err_str or "NOT_FOUND" in err_str:
                # Wrong model name for this SDK version — skip immediately
                logger.warning(
                    f"Model '{model}' not available in this SDK version — skipping"
                )
                return None

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Parse retry delay
                delay = 15
                try:
                    import re
                    match = re.search(r"retryDelay.*?(\d+)s", err_str)
                    if match:
                        delay = int(match.group(1)) + 2
                except Exception:
                    pass
                logger.warning(
                    f"Model '{model}' quota exceeded for "
                    f"{os.path.basename(filepath)}. "
                    f"Waiting {delay}s then trying next model..."
                )
                time.sleep(delay)
                return None

            logger.error(
                f"Vision model '{model}' error for "
                f"{os.path.basename(filepath)}: {e}"
            )
            return None

    def _load_image_ocr(self, filepath: str) -> str:
        """Tesseract OCR fallback."""
        try:
            import pytesseract
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)
            if text.strip():
                return text.strip()
        except ImportError:
            logger.error(
                "Tesseract not installed. "
                "Download: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {filepath}: {e}")

        # Last resort — store filename so chunk is not empty
        filename = os.path.basename(filepath)
        return (
            f"Image file: {filename}\n"
            f"Path: {filepath}\n"
            f"Note: Vision extraction unavailable. "
            f"Re-ingest this file when API quota resets."
        )

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    def _build_metadata(self, filepath: str, ext: str) -> dict:
        stat = os.stat(filepath)
        is_image = ext in {".png", ".jpg", ".jpeg", ".webp"}
        return {
            "source":     filepath,
            "filename":   os.path.basename(filepath),
            "extension":  ext,
            "size_bytes": stat.st_size,
            "file_type":  ext.lstrip("."),
            "is_image":   str(is_image),
        }