from __future__ import annotations

import fitz
from PIL import Image
import pytesseract

from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, SourceDocument
from src.config.settings import settings


class ReadPDFTool(BaseTool):
    name = "read_pdf"
    description = (
        "Extract text from a patient PDF file. "
        "Uses OCR fallback for scanned or handwritten pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the PDF file"}
        },
        "required": ["path"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        path = inputs["path"]
        try:
            doc = fitz.open(path)
        except Exception as e:
            src = SourceDocument(
                path=path,
                page_count=0,
                raw_text="",
                extraction_method="failed",
            )
            memory.source_documents.append(src)
            return f"[READ_PDF_FAILED] {path} — {str(e)}"

        pages_text: list[str] = []
        final_method = "pymupdf"

        for page in doc:
            text = page.get_text().strip()
            if len(text) < settings.pdf_ocr_fallback_min_chars:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    final_method = "ocr"
                else:
                    # OCR yielded nothing; keep whatever pymupdf found (may be empty)
                    final_method = "failed"
            pages_text.append(text)

        doc.close()
        full_text = "\n".join(pages_text)
        src = SourceDocument(
            path=path,
            page_count=len(pages_text),
            raw_text=full_text,
            extraction_method=final_method,
        )
        memory.source_documents.append(src)
        return f"[READ_PDF_OK] {path} — {len(full_text)} chars via {final_method}"

    def _ocr_page(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""
