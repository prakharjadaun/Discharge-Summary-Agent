from src.agents.tools.reconcile_medications import ReconcileMedicationsTool
from src.agents.shared_memory import SharedMemory, Medication


async def test_new_medication_no_reason_flagged():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    mem.admission_medications = [
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False)
    ]
    mem.discharge_medications = [
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False),
        Medication(name="TAB. EMESET", dose="4MG", frequency="1-1-1", reason_documented=False),
    ]
    result = await tool.execute({}, mem)
    assert "RECONCILIATION_NEEDED" in result
    flags = [f for f in mem.flags if f.severity == "RECONCILIATION_NEEDED"]
    assert len(flags) == 1
    assert "EMESET" in flags[0].reason


async def test_stopped_medication_no_reason_flagged():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    mem.admission_medications = [
        Medication(name="METFORMIN", dose="500MG", frequency="1-0-1", reason_documented=False),
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False),
    ]
    mem.discharge_medications = [
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False),
    ]
    result = await tool.execute({}, mem)
    flags = [f for f in mem.flags if "METFORMIN" in f.reason]
    assert len(flags) == 1
    assert flags[0].severity == "RECONCILIATION_NEEDED"


async def test_continued_medication_no_flag():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    med = Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False)
    mem.admission_medications = [med]
    mem.discharge_medications = [med]
    result = await tool.execute({}, mem)
    assert len(mem.flags) == 0
    assert "[RECONCILE_OK]" in result


async def test_empty_lists_no_flag():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[RECONCILE_OK]" in result
    assert len(mem.flags) == 0
