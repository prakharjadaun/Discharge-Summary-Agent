from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag

VALID_SEVERITIES = {"MISSING", "PENDING", "CONFLICT", "RECONCILIATION_NEEDED", "SAFETY"}


class FlagForClinicianReviewTool(BaseTool):
    name = "flag_for_clinician_review"
    description = "Escalate a field or finding for clinician review. Always succeeds."
    parameters = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "reason": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["MISSING", "PENDING", "CONFLICT", "RECONCILIATION_NEEDED", "SAFETY"],
            },
        },
        "required": ["field", "reason", "severity"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        field = inputs.get("field")
        reason = inputs.get("reason")
        severity = inputs.get("severity")

        if field is None or reason is None or severity is None:
            missing = [k for k, v in [("field", field), ("reason", reason), ("severity", severity)] if not v]
            return f"[FLAG_ERROR: missing required input(s): {', '.join(missing)}]"

        if severity not in VALID_SEVERITIES:
            return f"[FLAG_ERROR: invalid severity '{severity}' — must be one of {sorted(VALID_SEVERITIES)}]"

        flag = ClinicalFlag(
            field=field,
            reason=reason,
            severity=severity,
            source_docs=[],
        )
        memory.flags.append(flag)
        return f"[FLAGGED] {severity} — {field}: {reason}"
