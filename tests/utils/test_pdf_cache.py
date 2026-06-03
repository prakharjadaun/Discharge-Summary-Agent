import json
from pathlib import Path
import pytest
from src.utils.pdf_cache import PDFCache


def test_get_returns_none_for_uncached_file(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf content")
    cache = PDFCache(str(tmp_path / ".cache"))
    assert cache.get(str(pdf)) is None


def test_put_get_roundtrip(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf content")
    cache = PDFCache(str(tmp_path / ".cache"))
    pages = [{"page_num": 0, "text": "Hello world", "method": "pymupdf", "char_count": 11}]
    cache.put(str(pdf), pages)
    entry = cache.get(str(pdf))
    assert entry is not None
    assert entry["page_count"] == 1
    assert entry["pages"][0]["text"] == "Hello world"
    assert entry["pages"][0]["method"] == "pymupdf"
    assert "extracted_at" in entry
    assert "sha256" in entry


def test_cache_keyed_by_content_not_path(tmp_path):
    # Same bytes at different paths → same cache entry
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pdf_a = tmp_path / "a" / "doc.pdf"
    pdf_b = tmp_path / "b" / "doc.pdf"
    same_bytes = b"identical content"
    pdf_a.write_bytes(same_bytes)
    pdf_b.write_bytes(same_bytes)
    cache = PDFCache(str(tmp_path / ".cache"))
    pages = [{"page_num": 0, "text": "text", "method": "ocr", "char_count": 4}]
    cache.put(str(pdf_a), pages)
    assert cache.get(str(pdf_b)) is not None


def test_different_content_different_cache_entries(tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"content A")
    pdf_b.write_bytes(b"content B")
    cache = PDFCache(str(tmp_path / ".cache"))
    pages = [{"page_num": 0, "text": "A", "method": "pymupdf", "char_count": 1}]
    cache.put(str(pdf_a), pages)
    assert cache.get(str(pdf_b)) is None


def test_cache_creates_directory_if_missing(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"content")
    cache_dir = tmp_path / "deep" / "nested" / ".cache"
    cache = PDFCache(str(cache_dir))
    pages = [{"page_num": 0, "text": "hi", "method": "pymupdf", "char_count": 2}]
    cache.put(str(pdf), pages)
    assert cache_dir.exists()
    assert cache.get(str(pdf)) is not None
