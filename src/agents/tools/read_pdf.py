from __future__ import annotations

import asyncio
import logging
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
        path = inputs.get("path")
        if not path:
            return "[READ_PDF_FAILED: missing required input 'path']"

        pages_text: list[str] = []
        final_method = "pymupdf"

        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text = page.get_text().strip()
                    if len(text) < settings.pdf_ocr_fallback_min_chars:
                        ocr_text = await asyncio.get_event_loop().run_in_executor(
                            None, self._ocr_page, page
                        )
                        if ocr_text:
                            text = ocr_text
                            if final_method == "pymupdf":
                                final_method = "ocr"
                        else:
                            if final_method == "pymupdf":
                                final_method = "failed"
                    pages_text.append(text)
        except Exception as e:
            src = SourceDocument(
                path=path,
                page_count=0,
                raw_text="",
                extraction_method="failed",
            )
            memory.source_documents.append(src)
            return f"[READ_PDF_FAILED] {path} — {str(e)}"

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
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("OCR failed for page: %s", e)
            return ""
