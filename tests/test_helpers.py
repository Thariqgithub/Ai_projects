import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    generate_doc_id,
    get_file_extension,
    chunk_list,
    flatten,
    sanitize_filename,
    format_metadata,
)


def test_generate_doc_id_deterministic():
    assert generate_doc_id("hello world") == generate_doc_id("hello world")


def test_generate_doc_id_unique():
    assert generate_doc_id("hello") != generate_doc_id("world")


def test_get_file_extension():
    assert get_file_extension("report.pdf") == ".pdf"
    assert get_file_extension("data.CSV") == ".csv"
    assert get_file_extension("noext") == ""


def test_chunk_list():
    data = list(range(10))
    chunks = chunk_list(data, 3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_chunk_list_exact():
    data = list(range(6))
    chunks = chunk_list(data, 3)
    assert len(chunks) == 2


def test_flatten():
    nested = [[1, 2], [3, 4], [5]]
    assert flatten(nested) == [1, 2, 3, 4, 5]


def test_sanitize_filename():
    assert sanitize_filename("my file name!.pdf") == "my_file_name_.pdf"
    assert sanitize_filename("valid_name-123.txt") == "valid_name-123.txt"


def test_format_metadata():
    meta = {"source": "doc.txt", "page": 1}
    result = format_metadata(meta)
    assert "source: doc.txt" in result
    assert "page: 1" in result
