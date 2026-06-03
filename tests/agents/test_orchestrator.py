from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import DischargeAgentOrchestrator
from src.agents.critic_agent import CriticFeedback
from src.agents.shared_memory import SharedMemory


async def test_orchestrator_approves_on_first_review():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(return_value=CriticFeedback(approved=True))

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 3
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir="data/patient2", patient_id="p002")

    assert memory.critic_approved is True
    assert memory.handoff_round == 1
    executor.address_feedback.assert_not_called()


async def test_orchestrator_retries_after_feedback():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(side_effect=[
        CriticFeedback(approved=False, issues=["allergies missing"]),
        CriticFeedback(approved=True),
    ])

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 3
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir="data/patient2", patient_id="p002")

    assert memory.critic_approved is True
    assert memory.handoff_round == 2
    executor.address_feedback.assert_called_once_with(["allergies missing"], memory)


async def test_orchestrator_flags_when_never_approved():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(
        return_value=CriticFeedback(approved=False, issues=["unresolved issue"])
    )

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 2
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir=".", patient_id="p001")

    assert memory.critic_approved is False
    never_approved_flags = [f for f in memory.flags if "did not approve" in f.reason.lower()]
    assert len(never_approved_flags) == 1
