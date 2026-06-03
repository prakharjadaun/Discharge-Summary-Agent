from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, Conflict, ClinicalFlag


class DetectConflictsTool(BaseTool):
    name = "detect_conflicts"
    description = "Flag when two source documents disagree on the same clinical field."
    parameters = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "values": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map of document_name → value_from_that_doc",
            },
        },
        "required": ["field", "values"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        field = inputs.get("field")
        values = inputs.get("values")

        missing = []
        if not field:
            missing.append("'field'")
        if values is None:
            missing.append("'values'")
        if missing:
            return f"[DETECT_ERROR: missing required input(s): {', '.join(missing)}]"

        if not values:
            return f"[NO_CONFLICT] {field}: no sources provided"

        unique_vals = set(v.strip().lower() for v in values.values())
        if len(unique_vals) <= 1:
            return f"[NO_CONFLICT] {field}: all sources agree"

        conflict = Conflict(field=field, values=values)
        memory.conflicts.append(conflict)
        memory.flags.append(ClinicalFlag(
            field=field,
            reason=f"Conflict in '{field}': {values}",
            severity="CONFLICT",
            source_docs=list(values.keys()),
        ))
        return f"[CONFLICT] {field}: {values}"
