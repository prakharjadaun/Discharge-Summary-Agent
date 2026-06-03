from __future__ import annotations
import json
from pathlib import Path
from src.agents.base_agent import BaseAgent
from src.agents.shared_memory import SharedMemory, ClinicalFlag
from src.config.settings import settings

EXECUTOR_SYSTEM_PROMPT = """You are a clinical data extraction agent producing a discharge summary draft.

HARD RULES — violation is not permitted:
1. NEVER infer, assume, or fill in missing clinical facts
2. If a field is absent → call flag_for_clinician_review with severity MISSING
3. If a result is pending → call flag_for_clinician_review with severity PENDING
4. If two documents disagree → call detect_conflicts — do NOT pick one
5. Compare admission and discharge medications → call reconcile_medications

WORKFLOW (follow in order):
1. Call read_pdf for EVERY PDF file in the patient folder
2. Call extract_section for each required field:
   patient_demographics, admission_date, discharge_date, principal_diagnosis,
   secondary_diagnoses, hospital_course, procedures, discharge_medications,
   allergies, follow_up_instructions, pending_results, discharge_condition
3. Call reconcile_medications
4. Call drug_interaction_lookup with the discharge medication names
5. When complete, respond with the text COMPILE and nothing else

You are done when you have processed all PDFs and all 12 sections.
"""


class ExecutorAgent(BaseAgent):

    async def run(self, patient_dir: str, memory: SharedMemory) -> None:
        pdf_paths = list(Path(patient_dir).glob("*.pdf"))
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Patient folder: {patient_dir}\n"
                f"PDF files found: {[str(p) for p in pdf_paths]}\n"
                "Begin processing."
            )},
        ]
        step_budget = memory.step_count + settings.agent_max_steps
        await self._run_loop(messages, memory, step_budget)

    async def address_feedback(self, issues: list[str], memory: SharedMemory) -> None:
        draft_snapshot = json.dumps(memory.draft or {}, indent=2)[:3000]
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The critic identified these issues in your draft:\n"
                + "\n".join(f"- {issue}" for issue in issues)
                + f"\n\nCurrent draft:\n{draft_snapshot}\n\n"
                "Address each issue. Use flag_for_clinician_review if you cannot resolve it from the source documents."
            )},
        ]
        step_budget = memory.step_count + settings.agent_max_steps
        await self._run_loop(messages, memory, step_budget)

    async def _run_loop(self, messages: list[dict], memory: SharedMemory, step_budget: int) -> None:
        while memory.step_count < step_budget:
            content, tool_calls, _ = await self._provider.stream_complete(
                messages=messages,
                tools=self._registry.get_openai_definitions(),
                memory=memory,
                agent=self._name,
                action="plan_and_act",
                inputs={"step": memory.step_count},
            )
            memory.step_count += 1

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })
                tool_results = await self._dispatch_tool_calls(tool_calls, memory)
                messages.extend(tool_results)
            else:
                self._compile_draft(memory)
                return

        memory.flags.append(ClinicalFlag(
            field="agent_control",
            reason=f"Step cap ({settings.agent_max_steps}) reached — draft may be incomplete",
            severity="SAFETY",
            source_docs=[],
        ))
        self._compile_draft(memory)

    def _compile_draft(self, memory: SharedMemory) -> None:
        memory.draft = {
            section: field.model_dump()
            for section, field in memory.extracted_sections.items()
        }
