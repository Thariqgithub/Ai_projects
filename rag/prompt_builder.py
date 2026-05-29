from utils.helpers import format_metadata
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent assistant that answers questions based on the provided context. "
    "Always ground your answers in the retrieved context. "
    "If the answer cannot be found in the context, say so clearly. "
    "Be concise, accurate, and cite sources when possible."
)


class PromptBuilder:
    """
    Constructs RAG prompts from a query + retrieved context chunks.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_tokens: int = 6000,  # approximate token budget for context
        include_metadata: bool = True,
        include_chunk_index: bool = True,
    ):
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata
        self.include_chunk_index = include_chunk_index

    def build(self, query: str, chunks: list[dict]) -> dict:
        """
        Build a full prompt dict: {"system": ..., "user": ...}
        """
        context_str = self._build_context(chunks)
        user_prompt = self._build_user_prompt(query, context_str)
        return {
            "system": self.system_prompt,
            "user": user_prompt,
        }

    def _build_context(self, chunks: list[dict]) -> str:
        """Assemble context block from retrieved chunks."""
        parts = []
        char_budget = self.max_context_tokens * 4  # ~4 chars per token
        total_chars = 0

        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            meta = chunk.get("metadata", {})

            if total_chars + len(content) > char_budget:
                logger.debug(f"Context budget exceeded at chunk {i}, truncating.")
                break

            header = ""
            if self.include_chunk_index:
                header += f"[Chunk {i + 1}]"
            if self.include_metadata and meta:
                source = meta.get("filename") or meta.get("source", "")
                if source:
                    header += f" Source: {source}"

            chunk_str = f"{header}\n{content}" if header else content
            parts.append(chunk_str)
            total_chars += len(chunk_str)

        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:"
        )

    def build_conversation(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] = None,
    ) -> list[dict]:
        """
        Build a messages list for multi-turn conversation.
        history: list of {"role": "user"|"assistant", "content": ...}
        """
        messages = []
        if history:
            messages.extend(history)

        context_str = self._build_context(chunks)
        user_content = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages
