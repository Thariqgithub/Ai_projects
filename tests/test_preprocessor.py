import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.preprocess import TextPreprocessor


def make_doc(text):
    return {"content": text, "metadata": {}}


def test_remove_urls():
    proc = TextPreprocessor(remove_urls=True)
    doc = make_doc("Visit https://example.com for more info.")
    result = proc.process(doc)
    assert "https://example.com" not in result["content"]


def test_remove_emails():
    proc = TextPreprocessor(remove_emails=True)
    doc = make_doc("Contact us at support@example.com")
    result = proc.process(doc)
    assert "support@example.com" not in result["content"]


def test_normalize_whitespace():
    proc = TextPreprocessor(normalize_whitespace=True)
    doc = make_doc("Hello    world\n\n\n\nEnd")
    result = proc.process(doc)
    assert "    " not in result["content"]
    assert "\n\n\n" not in result["content"]


def test_lowercase():
    proc = TextPreprocessor(lowercase=True)
    doc = make_doc("Hello WORLD")
    result = proc.process(doc)
    assert result["content"] == "hello world"


def test_no_modification_when_flags_off():
    proc = TextPreprocessor(
        remove_urls=False,
        remove_emails=False,
        normalize_whitespace=False,
        lowercase=False,
    )
    original = "Hello World https://x.com test@x.com"
    doc = make_doc(original)
    result = proc.process(doc)
    assert "https://x.com" in result["content"]


def test_process_many():
    proc = TextPreprocessor()
    docs = [make_doc("Doc one https://a.com"), make_doc("Doc two test@b.com")]
    results = proc.process_many(docs)
    assert len(results) == 2
    assert "https://a.com" not in results[0]["content"]
    assert "test@b.com" not in results[1]["content"]
