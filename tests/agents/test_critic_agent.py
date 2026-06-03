from unittest.mock import AsyncMock, MagicMock
from src.agents.critic_agent import CriticAgent, CriticFeedback
from src.agents.shared_memory import SharedMemory, ExtractedField, TraceStep
from datetime import datetime, timezone


def make_step():
    return TraceStep(
        step_id=0, agent="critic", reasoning="", action="review",
        inputs={}, result="", tokens_in=10, tokens_out=5,
        latency_ms=50.0, timestamp=datetime.now(timezone.utc)
    )


async def test_critic_returns_approved():
    provider = MagicMock()
    provider.stream_complete = AsyncMock(return_value=("APPROVED", None, make_step()))

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {"principal_diagnosis": {"value": "Gastroenteritis", "confidence": "found"}}

    feedback = await agent.review(mem)
    assert feedback.approved is True
    assert feedback.issues == []


async def test_critic_returns_issues():
    provider = MagicMock()
    provider.stream_complete = AsyncMock(return_value=(
        "Issues found:\n- principal_diagnosis has no raw_quote\n- allergies section missing",
        None, make_step()
    ))

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {}

    feedback = await agent.review(mem)
    assert feedback.approved is False
    assert len(feedback.issues) >= 2


async def test_critic_calls_tools_before_verdict():
    tool_call = [{
        "id": "c1", "type": "function",
        "function": {"name": "flag_for_clinician_review",
                     "arguments": '{"field": "allergies", "reason": "missing", "severity": "MISSING"}'}
    }]
    call_count = 0
    async def stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("", tool_call, make_step())
        return ("APPROVED", None, make_step())

    provider = MagicMock()
    provider.stream_complete = stream_side_effect

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[FLAGGED]")

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {}

    feedback = await agent.review(mem)
    assert feedback.approved is True
    registry.dispatch.assert_called_once()
