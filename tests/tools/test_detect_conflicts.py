from src.agents.tools.detect_conflicts import DetectConflictsTool
from src.agents.shared_memory import SharedMemory


async def test_conflict_detected():
    tool = DetectConflictsTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({
        "field": "principal_diagnosis",
        "values": {
            "discharge_summary.pdf": "Acute Gastroenteritis",
            "er_chart.pdf": "DKA"
        }
    }, mem)
    assert "CONFLICT" in result
    assert len(mem.conflicts) == 1
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "CONFLICT"


async def test_no_conflict_when_values_agree():
    tool = DetectConflictsTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({
        "field": "discharge_condition",
        "values": {
            "note1.pdf": "Hemodynamically stable",
            "note2.pdf": "Hemodynamically stable"
        }
    }, mem)
    assert "NO_CONFLICT" in result
    assert len(mem.conflicts) == 0
    assert len(mem.flags) == 0


async def test_single_source_no_conflict():
    tool = DetectConflictsTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({
        "field": "allergies",
        "values": {"note1.pdf": "NKDA"}
    }, mem)
    assert "NO_CONFLICT" in result


async def test_missing_inputs_returns_error():
    tool = DetectConflictsTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[DETECT_ERROR" in result
