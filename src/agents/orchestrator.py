from __future__ import annotations
from src.agents.executor_agent import ExecutorAgent
from src.agents.critic_agent import CriticAgent
from src.agents.shared_memory import SharedMemory, ClinicalFlag, SourceDocument
from src.config.settings import settings


class DischargeAgentOrchestrator:

    def __init__(self, executor: ExecutorAgent, critic: CriticAgent):
        self._executor = executor
        self._critic = critic

    async def run(
        self,
        patient_dir: str,
        patient_id: str,
        pdf_files: list[str] | None = None,
        pre_documents: list[SourceDocument] | None = None,
    ) -> SharedMemory:
        memory = SharedMemory(patient_id=patient_id)

        if pre_documents:
            memory.source_documents.extend(pre_documents)

        await self._executor.run(patient_dir=patient_dir, memory=memory, pdf_files=pdf_files)

        for _ in range(settings.agent_max_handoff_rounds):
            memory.handoff_round += 1
            feedback = await self._critic.review(memory)

            if feedback.approved:
                memory.critic_approved = True
                break

            await self._executor.address_feedback(feedback.issues, memory)

        if not memory.critic_approved:
            memory.flags.append(ClinicalFlag(
                field="review_status",
                reason=(
                    f"Critic did not approve after {memory.handoff_round} handoff round(s) — "
                    "full clinician review required before use"
                ),
                severity="MISSING",
                source_docs=[],
            ))

        return memory
