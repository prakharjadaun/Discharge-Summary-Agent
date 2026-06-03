import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.executor_agent import ExecutorAgent
from src.agents.shared_memory import SharedMemory, SourceDocument, TraceStep
from datetime import datetime, timezone


def make_step():
    return TraceStep(
        step_id=0, agent="executor", reasoning="", action="plan",
        inputs={}, result="", tokens_in=10, tokens_out=5,
        latency_ms=100.0, timestamp=datetime.now(timezone.utc)
    )


def make_provider_sequence(responses: list):
    call_count = 0
    async def stream_complete(*args, **kwargs):
        nonlocal call_count
        content, tool_calls = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return content, tool_calls, make_step()
    provider = MagicMock()
    provider.stream_complete = stream_complete
    return provider


async def test_executor_compiles_draft_after_no_tool_call(tmp_path):
    read_tool_call = [{
        "id": "call_1", "type": "function",
        "function": {"name": "read_pdf", "arguments": json.dumps({"path": "test.pdf"})}
    }]

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[READ_PDF_OK] test.pdf — 500 chars via pymupdf")

    provider = make_provider_sequence([
        ("", read_tool_call),
        ("COMPILE", None),
    ])

    agent = ExecutorAgent(provider, registry, "executor")
    mem = SharedMemory(patient_id="p001")
    mem.source_documents.append(SourceDocument(
        path="test.pdf", page_count=1, raw_text="content", extraction_method="pymupdf"
    ))
    await agent.run(patient_dir=str(tmp_path), memory=mem)

    assert mem.draft is not None


async def test_executor_respects_step_cap():
    tool_call = [{"id": "c", "type": "function", "function": {"name": "read_pdf", "arguments": '{"path": "f.pdf"}'}}]

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[READ_PDF_OK]")

    responses = [("", tool_call)] * 25
    provider = make_provider_sequence(responses)

    with patch("src.agents.base_agent.settings") as ms:
        ms.agent_max_steps = 5
        agent = ExecutorAgent(provider, registry, "executor")
        mem = SharedMemory(patient_id="p001")
        await agent.run(patient_dir=".", memory=mem)

    cap_flags = [f for f in mem.flags if "cap" in f.reason.lower()]
    assert len(cap_flags) >= 1
