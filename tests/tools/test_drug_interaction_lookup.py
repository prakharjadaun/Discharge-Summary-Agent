from src.agents.tools.drug_interaction_lookup import DrugInteractionLookupTool
from src.agents.shared_memory import SharedMemory


async def test_known_interaction_flagged():
    tool = DrugInteractionLookupTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"medications": ["warfarin", "aspirin"]}, mem)
    assert "bleeding risk" in result.lower()
    safety_flags = [f for f in mem.flags if f.severity == "SAFETY"]
    assert len(safety_flags) == 1


async def test_no_interaction():
    tool = DrugInteractionLookupTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"medications": ["paracetamol", "vitamin_c"]}, mem)
    assert "no known interactions" in result.lower()
    assert len(mem.flags) == 0


async def test_missing_medications_input():
    tool = DrugInteractionLookupTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "no known interactions" in result.lower() or "[DRUG_LOOKUP" in result
