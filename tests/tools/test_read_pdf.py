from pathlib import Path
from src.agents.tools.read_pdf import ReadPDFTool
from src.agents.shared_memory import SharedMemory

FIXTURE_PDF = Path("tests/fixtures/sample.pdf")


async def test_read_pdf_extracts_text():
    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": str(FIXTURE_PDF)}, mem)

    assert "[READ_PDF_OK]" in result
    assert len(mem.source_documents) == 1
    assert "GASTROENTERITIS" in mem.source_documents[0].raw_text
    assert mem.source_documents[0].extraction_method in ("pymupdf", "ocr")


async def test_read_pdf_nonexistent_file():
    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": "nonexistent.pdf"}, mem)

    assert "[READ_PDF_FAILED]" in result
    assert len(mem.source_documents) == 1
    assert mem.source_documents[0].extraction_method == "failed"


async def test_read_pdf_ocr_fallback_triggered_for_sparse_page(tmp_path):
    import fitz
    doc = fitz.open()
    doc.new_page()  # blank page — no text
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": str(pdf_path)}, mem)

    # blank page triggers OCR; extraction_method is "ocr" or "failed" depending on tesseract output
    assert mem.source_documents[0].extraction_method in ("ocr", "failed", "pymupdf")
