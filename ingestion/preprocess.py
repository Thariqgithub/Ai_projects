import re
import unicodedata

from utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Cleans and normalizes raw text before chunking/embedding.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        remove_special_chars: bool = False,
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars

    def process(self, document: dict) -> dict:
        """Process a loaded document dict in-place and return it."""
        text = document["content"]
        text = self._clean(text)
        document["content"] = text
        return document

    def process_many(self, documents: list[dict]) -> list[dict]:
        return [self.process(doc) for doc in documents]

    def _clean(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # Remove null bytes and control characters (keep \n \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)

        if self.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal space
            text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 newlines
            text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text
