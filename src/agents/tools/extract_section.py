from __future__ import annotations
import json
from src.agents.tools.base import BaseTool
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.shared_memory import SharedMemory, ExtractedField, ClinicalFlag

EXTRACT_SYSTEM_PROMPT = """You are a clinical data extraction assistant.
Extract ONLY text that appears VERBATIM in the source document.
Never infer, assume, or fill gaps.

Return ONLY valid JSON:
{"confidence": "found"|"missing"|"pending", "value": <string or null>, "raw_quote": <verbatim snippet or null>}

- "found": section clearly stated in document. Include the exact quote.
- "pending": explicitly noted as pending/awaited.
- "missing": not present in the document at all.
"""


class ExtractSectionTool(BaseTool):
    @property
    def name(self) -> str:
        return "extract_section"

    @property
    def description(self) -> str:
        return "Extract a named clinical section from a source document. Never infers — returns missing if absent."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_path": {"type": "string"},
                "section_name": {"type": "string"},
            },
            "required": ["document_path", "section_name"],
        }

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        doc_path = inputs.get("document_path")
        section_name = inputs.get("section_name")

        if not doc_path or not section_name:
            return "[EXTRACT_ERROR: missing required inputs 'document_path' or 'section_name']"

        source = next((d for d in memory.source_documents if d.path == doc_path), None)
        if not source:
            return f"[EXTRACT_ERROR: document {doc_path} not loaded — call read_pdf first]"

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Document text:\n{source.raw_text[:6000]}\n\n"
                f"Extract section: {section_name}"
            )},
        ]

        content, _, _ = await self._provider.stream_complete(
            messages=messages, tools=None,
            memory=memory, agent="executor",
            action="extract_section",
            inputs={"section": section_name, "doc": doc_path},
        )

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = {"confidence": "missing", "value": None, "raw_quote": None}

        confidence = parsed.get("confidence", "missing")
        value = parsed.get("value")
        raw_quote = parsed.get("raw_quote")

        field = ExtractedField(
            value=value, source_doc=doc_path,
            confidence=confidence, raw_quote=raw_quote
        )
        memory.extracted_sections[section_name] = field

        if confidence == "missing":
            memory.flags.append(ClinicalFlag(
                field=section_name,
                reason=f"Section '{section_name}' not found in {doc_path}",
                severity="MISSING",
                source_docs=[doc_path],
            ))
            return f"[MISSING] {section_name} not found in {doc_path}"

        if confidence == "pending":
            memory.flags.append(ClinicalFlag(
                field=section_name,
                reason=f"Section '{section_name}' is pending: {value}",
                severity="PENDING",
                source_docs=[doc_path],
            ))
            return f"[PENDING] {section_name}: {value}"

        return f"[EXTRACTED] {section_name}: {value}"
