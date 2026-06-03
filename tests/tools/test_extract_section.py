from unittest.mock import AsyncMock, MagicMock
from src.agents.tools.extract_section import ExtractSectionTool
from src.agents.shared_memory import SharedMemory, SourceDocument, TraceStep
from datetime import datetime, timezone


def make_provider(content: str):
    provider = MagicMock()
    step = TraceStep(
        step_id=0, agent="executor", reasoning=content, action="extract_section",
        inputs={}, result=content, tokens_in=50, tokens_out=30,
        latency_ms=200.0, timestamp=datetime.now(timezone.utc)
    )
    provider.stream_complete = AsyncMock(return_value=(content, None, step))
    return provider


def make_memory_with_doc(text: str) -> SharedMemory:
    mem = SharedMemory(patient_id="p001")
    mem.source_documents.append(SourceDocument(
        path="discharge.pdf", page_count=1,
        raw_text=text, extraction_method="pymupdf"
    ))
    return mem


async def test_extract_section_found():
    llm_response = '{"confidence": "found", "value": "Acute Gastroenteritis", "raw_quote": "DIAGNOSIS: ACUTE GASTROENTERITIS"}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("DIAGNOSIS: ACUTE GASTROENTERITIS\nUTI")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "principal_diagnosis"}, mem
    )

    assert "[EXTRACTED]" in result
    assert "Acute Gastroenteritis" in result
    assert "principal_diagnosis" in mem.extracted_sections
    field = mem.extracted_sections["principal_diagnosis"]
    assert field.confidence == "found"
    assert field.value == "Acute Gastroenteritis"
    assert field.raw_quote == "DIAGNOSIS: ACUTE GASTROENTERITIS"


async def test_extract_section_missing():
    llm_response = '{"confidence": "missing", "value": null, "raw_quote": null}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("Some other content with no allergies listed")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "allergies"}, mem
    )

    assert "[MISSING]" in result
    field = mem.extracted_sections["allergies"]
    assert field.confidence == "missing"
    assert field.value is None
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "MISSING"


async def test_extract_section_pending():
    llm_response = '{"confidence": "pending", "value": "Urine culture — report awaited", "raw_quote": "urine culture and sensitivity sent- report awaited"}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("urine culture and sensitivity sent- report awaited")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "pending_results"}, mem
    )

    assert "[PENDING]" in result
    field = mem.extracted_sections["pending_results"]
    assert field.confidence == "pending"
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "PENDING"


async def test_extract_section_document_not_loaded():
    provider = make_provider('{"confidence": "found", "value": "x", "raw_quote": "x"}')
    tool = ExtractSectionTool(provider)
    mem = SharedMemory(patient_id="p001")  # no source documents

    result = await tool.execute(
        {"document_path": "missing.pdf", "section_name": "allergies"}, mem
    )
    assert "[EXTRACT_ERROR" in result
    assert "missing.pdf" in result


async def test_extract_section_missing_inputs():
    provider = make_provider('{"confidence": "found", "value": "x", "raw_quote": "x"}')
    tool = ExtractSectionTool(provider)
    mem = SharedMemory(patient_id="p001")

    result = await tool.execute({}, mem)
    assert "[EXTRACT_ERROR" in result


async def test_extract_section_unknown_confidence_treated_as_missing():
    llm_response = '{"confidence": "uncertain", "value": "something", "raw_quote": "quote"}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("some text")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "allergies"}, mem
    )

    assert "[MISSING]" in result
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "MISSING"
