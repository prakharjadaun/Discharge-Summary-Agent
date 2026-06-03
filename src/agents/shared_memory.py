from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class SourceDocument(BaseModel):
    path: str
    page_count: int
    raw_text: str
    extraction_method: str  # "pymupdf" | "ocr" | "failed"


class ExtractedField(BaseModel):
    value: str | None
    source_doc: str
    confidence: str          # "found" | "missing" | "pending"
    raw_quote: str | None    # verbatim text from source — anti-fabrication anchor


class ClinicalFlag(BaseModel):
    field: str
    reason: str
    severity: str            # "MISSING" | "PENDING" | "CONFLICT" | "RECONCILIATION_NEEDED" | "SAFETY"
    source_docs: list[str]


class Conflict(BaseModel):
    field: str
    values: dict[str, str]   # { "doc_name.pdf": "value from that doc" }


class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None
    duration: str | None = None
    reason_documented: bool = False


class TraceStep(BaseModel):
    step_id: int
    agent: str
    reasoning: str
    action: str
    inputs: dict
    result: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    timestamp: datetime


class SharedMemory(BaseModel):
    patient_id: str
    source_documents: list[SourceDocument] = []
    extracted_sections: dict[str, ExtractedField] = {}
    admission_medications: list[Medication] = []
    discharge_medications: list[Medication] = []
    conflicts: list[Conflict] = []
    flags: list[ClinicalFlag] = []
    draft: dict | None = None
    trace: list[TraceStep] = []
    step_count: int = 0
    handoff_round: int = 0
    critic_approved: bool = False
