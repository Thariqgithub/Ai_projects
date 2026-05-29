import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_doc_id(content: str) -> str:
    """Generate a deterministic document ID from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_file_extension(filepath: str) -> str:
    return Path(filepath).suffix.lower()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_size_mb(filepath: str) -> float:
    return os.path.getsize(filepath) / (1024 * 1024)


def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten(nested: list[list[Any]]) -> list[Any]:
    return [item for sublist in nested for item in sublist]


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying a function on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from filenames."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def format_metadata(metadata: dict) -> str:
    """Format metadata dict into a readable string."""
    return " | ".join(f"{k}: {v}" for k, v in metadata.items())
