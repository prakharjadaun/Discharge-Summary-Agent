import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from src.utils.output_writer import save_output
from src.agents.shared_memory import (
    SharedMemory, TraceStep, ClinicalFlag, Conflict
)


def _make_memory() -> SharedMemory:
    m = SharedMemory(patient_id="p_test")
    m.draft = {
        "patient_demographics": {
            "value": "John Doe, 45M",
            "confidence": "found",
            "source_doc": "test.pdf",
            "raw_quote": "John Doe",
        },
        "principal_diagnosis": {
            "value": None,
            "confidence": "missing",
            "source_doc": "",
            "raw_quote": None,
        },
    }
    m.trace = [
        TraceStep(
            step_id=1, agent="executor", reasoning="reading pdf", action="read_pdf",
            inputs={"path": "test.pdf"}, result="[READ_PDF_OK]",
            tokens_in=10, tokens_out=20, latency_ms=150.0,
            timestamp=datetime(2026, 6, 3, 10, 0, 0, tzinfo=timezone.utc),
        )
    ]
    m.flags = [
        ClinicalFlag(field="principal_diagnosis", reason="not found in docs", severity="MISSING", source_docs=[])
    ]
    m.conflicts = [
        Conflict(field="diagnosis", values={"a.pdf": "NSTEMI", "b.pdf": "ACS"})
    ]
    m.critic_approved = True
    m.handoff_round = 1
    m.step_count = 10
    return m


def test_save_output_creates_four_files(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    assert (out / "discharge_summary.md").exists()
    assert (out / "discharge_summary.json").exists()
    assert (out / "trace.json").exists()
    assert (out / "flags.json").exists()


def test_save_output_returns_patient_subdir(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    assert out.name == "p_test"
    assert out.parent == tmp_path


def test_discharge_summary_json_required_fields(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    data = json.loads((out / "discharge_summary.json").read_text(encoding="utf-8"))
    assert data["patient_id"] == "p_test"
    assert data["critic_approved"] is True
    assert data["handoff_rounds"] == 1
    assert data["step_count"] == 10
    assert "draft" in data
    assert "flags" in data
    assert "conflicts" in data
    assert "generated_at" in data


def test_trace_json_is_list_of_steps(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    assert isinstance(trace, list)
    assert len(trace) == 1
    assert trace[0]["step_id"] == 1
    assert trace[0]["agent"] == "executor"
    assert trace[0]["action"] == "read_pdf"


def test_flags_json_contains_flags_and_conflicts(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    data = json.loads((out / "flags.json").read_text(encoding="utf-8"))
    assert "flags" in data
    assert "conflicts" in data
    assert data["flags"][0]["severity"] == "MISSING"
    assert data["conflicts"][0]["field"] == "diagnosis"


def test_discharge_summary_md_contains_patient_id(tmp_path):
    m = _make_memory()
    out = save_output(m, output_dir=str(tmp_path))
    md = (out / "discharge_summary.md").read_text(encoding="utf-8")
    assert "p_test" in md
    assert "Critic approved" in md


def test_save_output_creates_nested_dirs(tmp_path):
    m = _make_memory()
    deep_dir = tmp_path / "a" / "b" / "c"
    out = save_output(m, output_dir=str(deep_dir))
    assert out.exists()
