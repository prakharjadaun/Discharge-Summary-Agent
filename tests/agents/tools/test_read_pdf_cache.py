import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.tools.read_pdf import ReadPDFTool
from src.agents.shared_memory import SharedMemory, SourceDocument
from src.utils.pdf_cache import PDFCache


def _make_cache_entry(path: str) -> dict:
    return {
        "sha256": "abc123",
        "path": path,
        "page_count": 2,
        "extracted_at": "2026-06-03T10:00:00Z",
        "pages": [
            {"page_num": 0, "text": "Page one text", "method": "pymupdf", "char_count": 13},
            {"page_num": 1, "text": "Page two text", "method": "vision", "char_count": 13},
        ],
    }


@pytest.mark.asyncio
async def test_cache_hit_skips_extraction():
    cache = MagicMock(spec=PDFCache)
    cache.get.return_value = _make_cache_entry("data/test.pdf")

    tool = ReadPDFTool(provider=None, cache=cache)
    memory = SharedMemory(patient_id="p1")

    with patch("fitz.open") as mock_open:
        result = await tool.execute({"path": "data/test.pdf"}, memory)

    mock_open.assert_not_called()
    assert "cache" in result
    assert len(memory.source_documents) == 1
    doc = memory.source_documents[0]
    assert doc.page_count == 2
    assert doc.extraction_method == "vision"  # worst method across pages
    assert "Page one text" in doc.raw_text


@pytest.mark.asyncio
async def test_cache_miss_writes_to_cache(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"fake")
    cache = MagicMock(spec=PDFCache)
    cache.get.return_value = None  # miss

    tool = ReadPDFTool(provider=None, cache=cache)
    memory = SharedMemory(patient_id="p1")

    fake_page = MagicMock()
    fake_page.number = 0
    fake_page.get_text.return_value = "Typed text on this page that is long enough to pass the threshold check"

    with patch("fitz.open") as mock_fitz:
        mock_fitz.return_value.__enter__.return_value = [fake_page]
        await tool.execute({"path": str(pdf)}, memory)

    cache.put.assert_called_once()
    call_args = cache.put.call_args
    assert call_args[0][0] == str(pdf)
    pages = call_args[0][1]
    assert isinstance(pages, list)
    assert pages[0]["page_num"] == 0
    assert pages[0]["method"] == "pymupdf"


@pytest.mark.asyncio
async def test_already_in_memory_skips_everything():
    cache = MagicMock(spec=PDFCache)
    tool = ReadPDFTool(provider=None, cache=cache)
    memory = SharedMemory(patient_id="p1")
    memory.source_documents.append(
        SourceDocument(path="data/test.pdf", page_count=1, raw_text="existing", extraction_method="pymupdf")
    )

    with patch("fitz.open") as mock_open:
        result = await tool.execute({"path": "data/test.pdf"}, memory)

    mock_open.assert_not_called()
    cache.get.assert_not_called()
    assert "already extracted" in result
    assert len(memory.source_documents) == 1  # no duplicate


@pytest.mark.asyncio
async def test_no_cache_configured_falls_through_to_extraction(tmp_path):
    """When cache=None, extraction runs normally without any cache calls."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"fake")
    tool = ReadPDFTool(provider=None, cache=None)
    memory = SharedMemory(patient_id="p1")

    fake_page = MagicMock()
    fake_page.number = 0
    fake_page.get_text.return_value = "Enough text here to pass the minimum character threshold easily"

    with patch("fitz.open") as mock_fitz:
        mock_fitz.return_value.__enter__.return_value = [fake_page]
        result = await tool.execute({"path": str(pdf)}, memory)

    assert "READ_PDF_OK" in result
    assert "(cache)" not in result
    assert len(memory.source_documents) == 1
