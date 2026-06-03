from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag


class ReconcileMedicationsTool(BaseTool):
    name = "reconcile_medications"
    description = "Compare admission vs discharge medications and flag undocumented changes."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        admission_names = {m.name.upper(): m for m in memory.admission_medications}
        discharge_names = {m.name.upper(): m for m in memory.discharge_medications}
        findings: list[str] = []

        for name, med in discharge_names.items():
            if name not in admission_names:
                memory.flags.append(ClinicalFlag(
                    field=f"medication:{med.name}",
                    reason=f"NEW medication '{med.name}' on discharge — no documented reason",
                    severity="RECONCILIATION_NEEDED",
                    source_docs=[d.path for d in memory.source_documents],
                ))
                findings.append(f"NEW:{med.name}")

        for name, med in admission_names.items():
            if name not in discharge_names:
                memory.flags.append(ClinicalFlag(
                    field=f"medication:{med.name}",
                    reason=f"STOPPED medication '{med.name}' — not on discharge list, no documented reason",
                    severity="RECONCILIATION_NEEDED",
                    source_docs=[d.path for d in memory.source_documents],
                ))
                findings.append(f"STOPPED:{med.name}")

        if not findings:
            return "[RECONCILE_OK] No undocumented medication changes found"
        return f"[RECONCILIATION_NEEDED] {'; '.join(findings)}"
