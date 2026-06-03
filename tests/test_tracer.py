import pytest
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
    with LLMCallTracer(
        memory=mem,
        agent="executor",
        action="read_pdf",
        inputs={"path": "file.pdf"}
    ) as tracer:
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
    with LLMCallTracer(memory=mem, agent="executor", action="plan", inputs={}) as tracer:
        step = await tracer.record_failure("timeout after 30s")

    assert "[LLM_CALL_FAILED]" in step.result
    assert mem.step_count == 1


async def test_tracer_calls_emit_callback():
    emitted = []
    async def fake_callback(step):
        emitted.append(step)

    mem = SharedMemory(patient_id="p001")
    with LLMCallTracer(
        memory=mem, agent="executor", action="plan",
        inputs={}, emit_callback=fake_callback
    ) as tracer:
        await tracer.record("reasoning", make_mock_response())

    assert len(emitted) == 1


async def test_tracer_record_failure_calls_emit_callback():
    emitted = []
    async def fake_callback(step):
        emitted.append(step)

    mem = SharedMemory(patient_id="p001")
    with LLMCallTracer(
        memory=mem, agent="executor", action="plan",
        inputs={}, emit_callback=fake_callback
    ) as tracer:
        await tracer.record_failure("network error")

    assert len(emitted) == 1
    assert "[LLM_CALL_FAILED]" in emitted[0].result


async def test_tracer_step_id_increments():
    mem = SharedMemory(patient_id="p001")
    with LLMCallTracer(memory=mem, agent="executor", action="step1", inputs={}) as tracer:
        step_a = await tracer.record("first", make_mock_response())
    with LLMCallTracer(memory=mem, agent="executor", action="step2", inputs={}) as tracer:
        step_b = await tracer.record("second", make_mock_response())

    assert step_a.step_id == 0
    assert step_b.step_id == 1
    assert mem.step_count == 2
