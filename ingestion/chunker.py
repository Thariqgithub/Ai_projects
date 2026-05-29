import re
from typing import Optional

from config.settings import settings
from utils.helpers import generate_doc_id
from utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Splits documents into overlapping chunks for embedding.
    Supports character-level, sentence-level, and recursive splitting.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        strategy: str = "recursive",  # recursive | sentence | fixed
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk(self, document: dict) -> list[dict]:
        """
        Chunk a loaded document dict.
        Returns list of chunk dicts with content and metadata.
        """
        content: str = document["content"]
        metadata: dict = document.get("metadata", {})

        if not content.strip():
            return []

        if self.strategy == "recursive":
            chunks = self._recursive_split(content)
        elif self.strategy == "sentence":
            chunks = self._sentence_split(content)
        else:
            chunks = self._fixed_split(content)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_doc_id(chunk_text + str(i))
            result.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "chunk_size": len(chunk_text),
                },
            })

        logger.debug(f"Chunked document '{metadata.get('filename', 'unknown')}' into {len(result)} chunks")
        return result

    def chunk_many(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    def _recursive_split(self, text: str) -> list[str]:
        """Split by paragraphs → sentences → words recursively."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_recursive(text, separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._fixed_split(text)

        sep = separators[0]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    sub = self._split_recursive(part, separators[1:])
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())

        return self._apply_overlap(chunks)

    def _sentence_split(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            candidate = current + " " + sent if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return self._apply_overlap(chunks)

    def _fixed_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if c]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Merge overlap between consecutive chunks."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = merged[-1][-self.chunk_overlap:]
            merged.append(prev_tail + " " + chunks[i])
        return merged
