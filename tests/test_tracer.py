import pytest
import time
from unittest.mock import MagicMock
from src.providers.base.tracer import LLMCallTracer
from src.agents.shared_memory import SharedMemory


def make_mock_response(content="test output", prompt_tokens=100, completion_tokens=50):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


async def test_tracer_records_step():
    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(
        memory=mem,
        agent="executor",
        action="read_pdf",
        inputs={"path": "file.pdf"}
    )
    tracer.__enter__()
    response = make_mock_response("some content", 100, 50)
    step = await tracer.record("reasoning text", response)

    assert mem.step_count == 1
    assert len(mem.trace) == 1
    assert step.agent == "executor"
    assert step.action == "read_pdf"
    assert step.reasoning == "reasoning text"
    assert step.tokens_in == 100
    assert step.tokens_out == 50
    assert step.latency_ms > 0


async def test_tracer_record_failure():
    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(memory=mem, agent="executor", action="plan", inputs={})
    tracer.__enter__()
    step = await tracer.record_failure("timeout after 30s")

    assert "[LLM_CALL_FAILED]" in step.result
    assert mem.step_count == 1


async def test_tracer_calls_emit_callback():
    emitted = []

    async def fake_callback(step):
        emitted.append(step)

    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(
        memory=mem, agent="executor", action="plan",
        inputs={}, emit_callback=fake_callback
    )
    tracer.__enter__()
    await tracer.record("reasoning", make_mock_response())
    assert len(emitted) == 1
