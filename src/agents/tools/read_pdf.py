from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

import fitz
from PIL import Image
import pytesseract

from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, SourceDocument
from src.config.settings import settings

if TYPE_CHECKING:
    from src.utils.pdf_cache import PDFCache

_log = logging.getLogger(__name__)

_MIN_CHARS = settings.pdf_ocr_fallback_min_chars
_METHOD_RANK = {"pymupdf": 0, "ocr": 1, "vision": 2, "failed": 3}


class ReadPDFTool(BaseTool):
    name = "read_pdf"
    description = (
        "Extract text from a patient PDF file. "
        "Uses OCR for scanned printed pages and LLM vision for handwritten content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the PDF file"}
        },
        "required": ["path"],
    }

    def __init__(self, provider=None, cache: "PDFCache | None" = None):
        self._provider = provider
        self._cache = cache

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        path = inputs.get("path")
        if not path:
            return "[READ_PDF_FAILED: missing required input 'path']"

        # Skip if already extracted into this memory (e.g. pre-populated)
        if any(d.path == path for d in memory.source_documents):
            return f"[READ_PDF_OK] {path} — already extracted, skipping"

        # Cache hit: return without re-extracting
        if self._cache is not None:
            entry = self._cache.get(path)
            if entry is not None:
                return self._load_from_cache(path, entry, memory)

        # Cache miss: run extraction cascade
        pages_text: list[str] = []
        page_results: list[dict] = []
        worst_method = "pymupdf"

        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text, method = await self._read_page(page)
                    pages_text.append(text)
                    page_results.append({
                        "page_num": page.number,
                        "text": text,
                        "method": method,
                        "char_count": len(text),
                    })
                    if _METHOD_RANK.get(method, 0) > _METHOD_RANK.get(worst_method, 0):
                        worst_method = method
        except Exception as e:
            memory.source_documents.append(SourceDocument(
                path=path, page_count=0, raw_text="", extraction_method="failed",
            ))
            return f"[READ_PDF_FAILED] {path} — {e}"

        if self._cache is not None:
            self._cache.put(path, page_results)

        full_text = "\n\n".join(pages_text)
        memory.source_documents.append(SourceDocument(
            path=path,
            page_count=len(pages_text),
            raw_text=full_text,
            extraction_method=worst_method,
        ))
        return f"[READ_PDF_OK] {path} — {len(full_text)} chars via {worst_method}"

    def _load_from_cache(self, path: str, entry: dict, memory: SharedMemory) -> str:
        pages = entry.get("pages", [])
        full_text = "\n\n".join(p["text"] for p in pages)
        worst_method = max(
            (p.get("method", "pymupdf") for p in pages),
            key=lambda m: _METHOD_RANK.get(m, 0),
            default="pymupdf",
        )
        memory.source_documents.append(SourceDocument(
            path=path,
            page_count=entry.get("page_count", len(pages)),
            raw_text=full_text,
            extraction_method=worst_method,
        ))
        return f"[READ_PDF_OK] {path} — {len(full_text)} chars via {worst_method} (cache)"

    async def _read_page(self, page) -> tuple[str, str]:
        text = page.get_text().strip()
        if len(text) >= _MIN_CHARS:
            return text, "pymupdf"

        ocr_text = await asyncio.get_running_loop().run_in_executor(None, self._ocr_page, page)
        if len(ocr_text) >= _MIN_CHARS:
            return ocr_text, "ocr"

        if self._provider is not None:
            vision_text = await self._vision_page(page)
            if vision_text and len(vision_text) >= _MIN_CHARS:
                return vision_text, "vision"

        best = ocr_text or text or ""
        return best, "failed" if not best else "ocr"

    def _ocr_page(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            _log.warning("Tesseract OCR failed: %s", e)
            return ""

    async def _vision_page(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image_b64 = base64.b64encode(img_bytes).decode()
            return await self._provider.vision_transcribe(image_b64)
        except Exception as e:
            _log.warning("Vision transcription failed: %s", e)
            return ""
