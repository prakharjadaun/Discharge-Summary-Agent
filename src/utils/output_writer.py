from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agents.shared_memory import SharedMemory

_REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]

_SEVERITY_ICON = {
    "MISSING": "⚠️", "PENDING": "⏳", "CONFLICT": "🔴",
    "RECONCILIATION_NEEDED": "💊", "SAFETY": "🚨",
}

_CONFIDENCE_BADGE = {
    "found": "✓ found", "pending": "⏳ pending", "missing": "⚠ missing",
}


def save_output(memory: SharedMemory, output_dir: str = "output") -> Path:
    out = Path(output_dir) / memory.patient_id
    out.mkdir(parents=True, exist_ok=True)

    _write_summary_md(memory, out)
    _write_summary_json(memory, out)
    _write_trace_json(memory, out)
    _write_flags_json(memory, out)

    return out


def _write_summary_md(memory: SharedMemory, out: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    approved = "Yes ✓" if memory.critic_approved else "No ⚠️ — full clinician review required"
    lines = [
        f"# Discharge Summary — {memory.patient_id}",
        f"Generated: {now} | Critic approved: {approved}",
        "",
        "---",
        "",
    ]
    sections = memory.draft or {}
    for key in _REQUIRED_SECTIONS:
        field = sections.get(key) or {}
        if isinstance(field, dict):
            value = field.get("value") or "**[MISSING — flagged for clinician review]**"
            confidence = field.get("confidence", "missing")
        else:
            value = "**[MISSING — flagged for clinician review]**"
            confidence = "missing"
        badge = _CONFIDENCE_BADGE.get(confidence, "?")
        label = key.replace("_", " ").title()
        lines += [f"## {label}  `{badge}`", "", value, ""]

    if memory.flags:
        lines += ["## Clinical Flags", ""]
        for f in memory.flags:
            icon = _SEVERITY_ICON.get(f.severity, "⚠️")
            lines.append(f"- {icon} **{f.severity}** | `{f.field}` — {f.reason}")
        lines.append("")

    (out / "discharge_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_json(memory: SharedMemory, out: Path) -> None:
    data = {
        "patient_id": memory.patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "critic_approved": memory.critic_approved,
        "handoff_rounds": memory.handoff_round,
        "step_count": memory.step_count,
        "draft": memory.draft or {},
        "flags": [f.model_dump() for f in memory.flags],
        "conflicts": [c.model_dump() for c in memory.conflicts],
    }
    (out / "discharge_summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _write_trace_json(memory: SharedMemory, out: Path) -> None:
    trace = [s.model_dump() for s in memory.trace]
    (out / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _write_flags_json(memory: SharedMemory, out: Path) -> None:
    data = {
        "flags": [f.model_dump() for f in memory.flags],
        "conflicts": [c.model_dump() for c in memory.conflicts],
    }
    (out / "flags.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
