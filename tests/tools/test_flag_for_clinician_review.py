from src.agents.tools.flag_for_clinician_review import FlagForClinicianReviewTool
from src.agents.shared_memory import SharedMemory


async def test_flag_is_written_to_memory():
    tool = FlagForClinicianReviewTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({
        "field": "discharge_date",
        "reason": "Not found in any document",
        "severity": "MISSING",
    }, mem)
    assert "[FLAGGED]" in result
    assert len(mem.flags) == 1
    assert mem.flags[0].field == "discharge_date"
    assert mem.flags[0].severity == "MISSING"


async def test_all_severity_values_accepted():
    tool = FlagForClinicianReviewTool()
    for severity in ["MISSING", "PENDING", "CONFLICT", "RECONCILIATION_NEEDED", "SAFETY"]:
        mem = SharedMemory(patient_id="p001")
        result = await tool.execute({"field": "test", "reason": "test", "severity": severity}, mem)
        assert "[FLAGGED]" in result
        assert mem.flags[0].severity == severity


async def test_missing_inputs_returns_error():
    tool = FlagForClinicianReviewTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[FLAG_ERROR" in result
    assert len(mem.flags) == 0
