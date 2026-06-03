import pytest
from src.agents.shared_memory import (
    SharedMemory, ExtractedField, ClinicalFlag, Medication,
    Conflict, SourceDocument, TraceStep
)
from datetime import datetime

def test_shared_memory_defaults():
    mem = SharedMemory(patient_id="p001")
    assert mem.patient_id == "p001"
    assert mem.step_count == 0
    assert mem.handoff_round == 0
    assert mem.critic_approved is False
    assert mem.draft is None
    assert mem.flags == []
    assert mem.conflicts == []
    assert mem.trace == []

def test_extracted_field_confidence_values():
    field = ExtractedField(
        value="Acute Gastroenteritis",
        source_doc="discharge.pdf",
        confidence="found",
        raw_quote="DIAGNOSIS: ACUTE GASTROENTERITIS"
    )
    assert field.confidence == "found"
    assert field.raw_quote is not None

def test_clinical_flag_severities():
    flag = ClinicalFlag(
        field="primary_diagnosis",
        reason="Conflict between ER chart and discharge summary",
        severity="CONFLICT",
        source_docs=["er_chart.pdf", "discharge.pdf"]
    )
    assert flag.severity == "CONFLICT"

def test_medication_model():
    med = Medication(
        name="TAB. RACIPER",
        dose="40MG",
        frequency="1-0-0",
        duration="7 DAYS",
        reason_documented=False
    )
    assert med.reason_documented is False

def test_trace_step_model():
    step = TraceStep(
        step_id=1,
        agent="executor",
        reasoning="Need to read admission notes first",
        action="read_pdf",
        inputs={"path": "data/patient2/admission.pdf"},
        result="[READ_PDF_OK] 1200 chars",
        tokens_in=150,
        tokens_out=80,
        latency_ms=342.5,
        timestamp=datetime.utcnow()
    )
    assert step.agent == "executor"
    assert step.tokens_in == 150

def test_source_document_model():
    doc = SourceDocument(
        path="discharge.pdf",
        page_count=3,
        raw_text="PATIENT: Jane Doe\nDIAGNOSIS: UTI",
        extraction_method="pymupdf"
    )
    assert doc.extraction_method == "pymupdf"

def test_conflict_model():
    conflict = Conflict(
        field="principal_diagnosis",
        values={"er_chart.pdf": "DKA", "discharge_summary.pdf": "Acute Gastroenteritis"}
    )
    assert len(conflict.values) == 2

def test_shared_memory_mutable_defaults_are_independent():
    mem1 = SharedMemory(patient_id="p001")
    mem2 = SharedMemory(patient_id="p002")
    mem1.flags.append(ClinicalFlag(field="x", reason="y", severity="MISSING", source_docs=[]))
    assert len(mem2.flags) == 0  # Pydantic v2 handles this correctly
