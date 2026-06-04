# Discharge Summary Agent — Dscribe Take-Home

An agentic AI system that reads raw patient source-note PDFs and produces structured discharge summary drafts for clinician review. Built from scratch without LangGraph or CrewAI.

---

## Architecture

```
PDF Upload (Chainlit)
       │
       ▼
Phase 1 — Pre-Extraction
  ReadPDFTool: PyMuPDF → Tesseract OCR → GPT-4o Vision (per page cascade)
  PDFCache: SHA256-keyed JSON cache — vision runs once per unique PDF
       │
       ▼
Phase 2 — Agent Loop
  ExecutorAgent  →  reasons, calls tools, builds draft
  CriticAgent    →  reviews completeness & conflicts
  Handoff loop (max 3 rounds) until critic approves
       │
       ▼
Output saved to output/{patient_id}/
  discharge_summary.md / .json  |  trace.json  |  flags.json
```

**Key design decisions:**
- Abstract `BaseLLMProvider` — swap Azure ↔ OpenAI with zero agent code changes
- Real agent loop (plan → tool → observe → re-plan), not a hardcoded pipeline
- No fabrication guardrail: any field not sourced from documents is marked `[MISSING]` and flagged
- Hard step cap (`agent_max_steps=50`) so the agent cannot loop forever
- Per-step structured trace: reasoning → tool → inputs → result → tokens → latency

---

## Setup

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on PATH
- Azure OpenAI access with a GPT-4o deployment

### Install

```bash
# Create and activate a virtual environment
python -m venv healthcare_venv
healthcare_venv\Scripts\activate        # Windows
# source healthcare_venv/bin/activate   # macOS/Linux

# Install the project
pip install -e .

# Install dev dependencies (for tests)
pip install -e ".[dev]"
```

### Configure

Copy `.env.example` to `.env` and fill in your Azure credentials:

```bash
cp .env.example .env
```

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_DEPLOYMENT=gpt-4o
AZURE_LLM_API_VERSION=2025-01-01-preview
AZURE_OPENAI_API_KEY=your-key-here
```

---

## Running

### Chainlit UI (recommended)

```bash
chainlit run chainlit_app.py
```

Open `http://localhost:8000`, upload a patient PDF using the 📎 attachment button.

**First run:** Pre-extracts all pages (vision calls for handwritten content), then runs the agent loop.  
**Subsequent runs:** Extraction is instant from cache — only agent LLM calls are made.

### Headless (CLI)

```bash
python -m src.agents.discharge_agent --patient-dir data/patient1/
```

---

## Output

Each run saves to `output/{patient_id}/`:

| File | Contents |
|------|----------|
| `discharge_summary.md` | Human-readable summary with confidence badges |
| `discharge_summary.json` | Structured draft + metadata (critic approved, handoff rounds, step count) |
| `trace.json` | Every agent step — reasoning, tool, inputs, result, tokens, latency |
| `flags.json` | Clinical flags (MISSING / PENDING / CONFLICT) + source documents read |

---

## Agent Tools

| Tool | Purpose |
|------|---------|
| `read_pdf` | Extract text — PyMuPDF → OCR → GPT-4o vision cascade |
| `extract_section` | Pull a specific clinical section from extracted text |
| `reconcile_medications` | Compare admission vs discharge meds, surface undocumented changes |
| `detect_conflicts` | Flag contradictions between source documents |
| `drug_interaction_lookup` | Mock external drug interaction check |
| `flag_for_clinician_review` | Escalation action — marks field as MISSING/PENDING/CONFLICT |
| `validate_completeness` | Critic tool — checks all 12 required sections are present |

---

## Tests

```bash
pytest
```

81 tests covering: settings, shared memory, all tools, executor/critic/orchestrator agents, PDF cache, output writer.

---

## Configuration

All tunables in `src/config/settings.py` (overridable via `.env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `agent_max_steps` | 50 | Hard cap on agent loop iterations |
| `agent_max_handoff_rounds` | 3 | Max executor↔critic handoff cycles |
| `pdf_ocr_fallback_min_chars` | 50 | Min chars before escalating to next extraction tier |
| `pdf_max_pages` | 5 | Max pages to extract per PDF (demo limit) |

---

## Project Structure

```
src/
  agents/
    executor_agent.py       — tool-calling agent loop
    critic_agent.py         — completeness & safety reviewer
    orchestrator.py         — handoff coordinator
    shared_memory.py        — Pydantic state model
    tools/                  — all agent tools
    tool_registry.py
  providers/
    base/llm_provider.py    — abstract LLM interface
    azure/llm_provider.py   — Azure OpenAI implementation
  config/settings.py        — Pydantic BaseSettings from .env
  utils/
    pdf_cache.py            — SHA256-keyed extraction cache
    output_writer.py        — save submission artefacts
chainlit_app.py             — Chainlit UI + pre-extraction phase
tests/                      — 81 tests
```

---

## Limitations & What's Next

- **Duplicate flags:** The same missing field gets flagged by executor, critic, and each handoff round — deduplication not yet implemented
- **Page limit:** `pdf_max_pages=5` for demo speed; increase for full clinical coverage
- **No cache expiry:** Cache entries never expire — delete `.pdf_cache/` to force re-extraction
- **Part 2 (Learning from Doctor Edits):** Not implemented — would use simulated reviewer + edit-distance reward signal to improve future drafts

---

## What to Submit

- [x] Source code (this repo)
- [x] Generated discharge summary drafts + step traces (`output/` — run the agent to generate)
- [x] Working video demo
- [ ] Part 2 learning loop (stretch — not attempted)
