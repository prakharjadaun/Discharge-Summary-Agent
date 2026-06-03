from unittest.mock import AsyncMock, MagicMock
from src.agents.tools.validate_completeness import ValidateCompletenessTool
from src.agents.shared_memory import SharedMemory, ExtractedField, TraceStep
from datetime import datetime, timezone


def make_provider(content: str):
    provider = MagicMock()
    step = TraceStep(
        step_id=0, agent="critic", reasoning=content, action="validate",
        inputs={}, result=content, tokens_in=50, tokens_out=30,
        latency_ms=100.0, timestamp=datetime.now(timezone.utc)
    )
    provider.stream_complete = AsyncMock(return_value=(content, None, step))
    return provider


async def test_validate_returns_issues_when_fields_missing():
    provider = make_provider('["principal_diagnosis is marked found but raw_quote is null"]')
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    mem.extracted_sections["principal_diagnosis"] = ExtractedField(
        value="Gastroenteritis", source_doc="note.pdf",
        confidence="found", raw_quote=None
    )
    result = await tool.execute({}, mem)
    assert "principal_diagnosis" in result


async def test_validate_returns_ok_when_all_clear():
    provider = make_provider('[]')
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[VALIDATION_OK]" in result


async def test_validate_handles_bad_json_gracefully():
    provider = make_provider("not valid json")
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[VALIDATION_OK]" in result  # bad JSON → empty issues → OK


async def test_validate_writes_flags_for_issues():
    provider = make_provider('["principal_diagnosis is marked found but raw_quote is null"]')
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    await tool.execute({}, mem)
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "MISSING"
    assert "principal_diagnosis" in mem.flags[0].reason
