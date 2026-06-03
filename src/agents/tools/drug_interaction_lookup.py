from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag

KNOWN_INTERACTIONS: dict[frozenset, str] = {
    frozenset(["warfarin", "aspirin"]): "Increased bleeding risk",
    frozenset(["warfarin", "ibuprofen"]): "Increased bleeding risk",
    frozenset(["metformin", "contrast_dye"]): "Lactic acidosis risk",
    frozenset(["ssri", "tramadol"]): "Serotonin syndrome risk",
    frozenset(["ace_inhibitor", "potassium"]): "Hyperkalaemia risk",
}


class DrugInteractionLookupTool(BaseTool):
    name = "drug_interaction_lookup"
    description = "Check discharge medications for known drug interactions. Returns warnings if found."
    parameters = {
        "type": "object",
        "properties": {
            "medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of medication names (lowercase)",
            }
        },
        "required": ["medications"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        try:
            meds = [m.lower().strip() for m in inputs.get("medications", [])]
            warnings: list[str] = []
            for pair, warning in KNOWN_INTERACTIONS.items():
                if pair.issubset(set(meds)):
                    warnings.append(f"{' + '.join(sorted(pair))}: {warning}")
                    memory.flags.append(ClinicalFlag(
                        field="drug_interaction",
                        reason=warning,
                        severity="SAFETY",
                        source_docs=[],
                    ))
            if not warnings:
                return "[DRUG_LOOKUP_OK] No known interactions found"
            return f"[DRUG_INTERACTION_WARNING] {'; '.join(warnings)}"
        except Exception as exc:
            return f"[DRUG_LOOKUP_UNAVAILABLE] {exc}"
