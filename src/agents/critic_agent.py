from __future__ import annotations
import json
from dataclasses import dataclass, field
from src.agents.base_agent import BaseAgent
from src.agents.shared_memory import SharedMemory

CRITIC_SYSTEM_PROMPT = """You are a clinical safety reviewer. The Executor has produced a discharge summary draft.

YOUR CHECKS:
1. Fabrication: any field with confidence="found" but raw_quote is null → flag it
2. Required sections completely absent from the draft → flag each
3. Medication reconciliation flags not raised for undocumented changes
4. Conflicts between documents not flagged
5. Drug interaction safety warnings not surfaced

If issues found: use detect_conflicts or flag_for_clinician_review for each issue, then
list ALL issues as bullet points starting with "Issues found:"

If everything is acceptable: respond with exactly "APPROVED" and nothing else.

You are the last line of defence before this draft reaches a clinician. Be strict.
"""

REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]


@dataclass
class CriticFeedback:
    approved: bool
    issues: list[str] = field(default_factory=list)


class CriticAgent(BaseAgent):

    async def review(self, memory: SharedMemory) -> CriticFeedback:
        draft_text = json.dumps(memory.draft or {}, indent=2)[:4000]
        flags_text = json.dumps([f.model_dump() for f in memory.flags], indent=2)[:2000]

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Draft:\n{draft_text}\n\n"
                f"Existing flags:\n{flags_text}\n\n"
                f"Required sections: {REQUIRED_SECTIONS}"
            )},
        ]

        while not self._step_cap_reached(memory):
            content, tool_calls, _ = await self._provider.stream_complete(
                messages=messages,
                tools=self._registry.get_openai_definitions(),
                memory=memory,
                agent=self._name,
                action="review",
                inputs={"handoff_round": memory.handoff_round},
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
                approved = content.strip().upper() == "APPROVED"
                if approved:
                    return CriticFeedback(approved=True)
                issues = [
                    line.lstrip("-• ").strip()
                    for line in content.splitlines()
                    if line.strip() and not line.strip().lower().startswith("issues found")
                ]
                return CriticFeedback(approved=False, issues=issues)

        return CriticFeedback(
            approved=False,
            issues=["Step cap reached during critic review — manual review required"]
        )
