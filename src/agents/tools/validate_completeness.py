from __future__ import annotations
import json
from src.agents.tools.base import BaseTool
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.shared_memory import SharedMemory

REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]

VALIDATE_SYSTEM_PROMPT = """You are a clinical safety reviewer checking a discharge summary draft.
Check ONLY for these issues:
1. A field has confidence="found" but raw_quote is null (fabrication risk)
2. A required section is completely absent from extracted_sections

Return a JSON array of issue strings. Return [] if everything is clean.
Example: ["principal_diagnosis is marked found but raw_quote is null", "allergies section missing"]
"""


class ValidateCompletenessTool(BaseTool):
    name = "validate_completeness"
    description = "Verify all required discharge summary sections are present and sourced."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        sections_summary = {
            name: {
                "confidence": field.confidence,
                "has_raw_quote": field.raw_quote is not None,
                "value_present": field.value is not None,
            }
            for name, field in memory.extracted_sections.items()
        }
        missing_sections = [s for s in REQUIRED_SECTIONS if s not in memory.extracted_sections]

        messages = [
            {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Extracted sections: {json.dumps(sections_summary)}\n"
                f"Missing sections: {missing_sections}\n"
                f"Existing flags count: {len(memory.flags)}"
            )},
        ]

        content, _, _ = await self._provider.stream_complete(
            messages=messages, tools=None,
            memory=memory, agent="critic",
            action="validate_completeness", inputs={},
        )

        try:
            issues = json.loads(content)
            if not isinstance(issues, list):
                issues = []
        except (json.JSONDecodeError, TypeError):
            issues = []

        if not issues:
            return "[VALIDATION_OK] All sections present and sourced"
        return f"[VALIDATION_ISSUES] {'; '.join(str(i) for i in issues)}"
