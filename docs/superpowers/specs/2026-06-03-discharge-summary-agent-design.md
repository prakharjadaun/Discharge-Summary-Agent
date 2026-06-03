# Discharge Summary Agent — Design Spec
**Date:** 2026-06-03
**Scope:** Part 1 only (Part 2 learning loop planned separately after Part 1 ships)
**Assignment:** Dscribe Take-Home — Agentic AI for Discharge Summaries

---

## Overview

An agentic AI system that reads a patient's raw source-note PDFs and produces a structured, clinically safe discharge summary draft for clinician review. The system uses a two-agent architecture (Executor + Critic) with shared memory, a class-based tool registry, full LLM call instrumentation, and a Chainlit UI for demo.

The primary design constraint is **clinical safety**: the agent must never fabricate or infer clinical facts. Every field in the output is either sourced (with a verbatim quote anchor) or explicitly marked `[MISSING]` / `[PENDING]` and flagged for clinician review.

---

## Architecture

```
chainlit_app.py
    └── DischargeAgentOrchestrator
            ├── ExecutorAgent  ──► ExecutorToolRegistry
            │       tools: ReadPDFTool, ExtractSectionTool,
            │               ReconcileMedicationsTool, DetectConflictsTool,
            │               DrugInteractionLookupTool, FlagForClinicianReviewTool
            │
            ├── CriticAgent   ──► CriticToolRegistry
            │       tools: DetectConflictsTool, ValidateCompletenessTool,
            │               FlagForClinicianReviewTool
            │
            └── SharedMemory  (read/write by both agents)
                    fields: source_documents, extracted_sections,
                            medications, conflicts, flags, draft,
                            trace, step_count, handoff_round, critic_approved

AzureLLMProvider (implements BaseLLMProvider)
    └── LLMCallTracer (wraps every LLM call — tokens, latency, reasoning)
```

---

## Project Structure

```
project/
    src/
        config/
            settings.py              # Pydantic BaseSettings from .env
        providers/
            base/
                llm_provider.py      # Abstract base class
                tracer.py            # LLMCallTracer context manager
            azure/
                llm_provider.py      # Azure OpenAI implementation
        agents/
            shared_memory.py         # SharedMemory + all data models
            base_agent.py            # BaseAgent (shared loop utilities)
            executor_agent.py        # ExecutorAgent
            critic_agent.py          # CriticAgent + CriticFeedback
            orchestrator.py          # DischargeAgentOrchestrator (handoff loop)
            tool_registry.py         # ToolRegistry class
            tools/
                base.py              # BaseTool ABC
                read_pdf.py          # ReadPDFTool
                extract_section.py   # ExtractSectionTool (needs LLM)
                reconcile_medications.py
                detect_conflicts.py
                drug_interaction_lookup.py   # mocked
                flag_for_clinician_review.py
                validate_completeness.py     # CriticAgent only (needs LLM)
                __init__.py          # build_executor_registry(), build_critic_registry()
    chainlit_app.py
    pyproject.toml
    .env                             # never committed
    .gitignore
    .dockerignore
    Dockerfile
    CLAUDE.md
```

---

## Component 1: Shared Memory + Data Models

**File:** `src/agents/shared_memory.py`

All state is centralised in `SharedMemory`, a Pydantic model passed by reference. Both agents read and write to it. No copying, no serialisation mid-run.

### Key models

```
SourceDocument       path, page_count, raw_text, extraction_method
ExtractedField       value, source_doc, confidence, raw_quote
ClinicalFlag         field, reason, severity, source_docs
Conflict             field, values: dict[doc → value]
Medication           name, dose, frequency, duration, reason_documented
TraceStep            step_id, agent, reasoning, action, inputs, result,
                     tokens_in, tokens_out, latency_ms, timestamp
SharedMemory         source_documents, extracted_sections,
                     admission_medications, discharge_medications,
                     conflicts, flags, draft, trace,
                     step_count, handoff_round, critic_approved
```

**Anti-fabrication anchor:** `ExtractedField.raw_quote` stores the verbatim text snippet from the source PDF. The Critic audits every field: if `value` diverges from `raw_quote`, or `raw_quote` is null, the field is challenged. `confidence` is one of `found | missing | pending`.

**ClinicalFlag severities:** `MISSING | PENDING | CONFLICT | RECONCILIATION_NEEDED | SAFETY`

---

## Component 2: Config + Provider Layer

**Files:** `src/config/settings.py`, `src/providers/base/llm_provider.py`, `src/providers/base/tracer.py`, `src/providers/azure/llm_provider.py`

### Settings (Pydantic BaseSettings)
Reads from `.env`. Fails fast on startup if any required field is missing.

```
azure_openai_endpoint       str
azure_llm_deployment        str       (gpt-4o)
azure_llm_api_version       str       (2025-01-01-preview)
azure_openai_api_key        str
agent_max_steps             int = 20  (hard cap across both agents)
agent_max_handoff_rounds    int = 3   (critic↔executor ping-pong limit)
pdf_ocr_fallback_min_chars  int = 50  (pages below this trigger OCR)
```

### BaseLLMProvider (abstract)
Three methods — agents never import openai directly:
- `complete(messages, tools, memory, agent, action, inputs) → (content, TraceStep)`
- `complete_with_tool_choice(messages, tools, memory, agent, action, inputs) → (content, tool_calls, TraceStep)`
- `stream_complete(messages, tools, memory, agent, action, inputs, token_callback?) → (content, tool_calls | None, TraceStep)`

`stream_complete` is the primary method used by both agents. `complete` and `complete_with_tool_choice` delegate to it internally.

### Streaming behaviour
All LLM calls use `stream=True` with `stream_options={"include_usage": True}` (Azure OpenAI returns a final usage chunk even in streaming mode — accurate token counts with no estimation needed).

- **Text responses** (Critic feedback, reasoning narration): tokens emitted chunk-by-chunk to Chainlit via `token_callback(chunk: str)` → `cl.Step.stream_token()`. Reasoning appears live in the UI.
- **Tool call responses**: tool call arguments accumulate from chunks before dispatch — a tool cannot be called with partial arguments. Tool name appears immediately in the Chainlit step; arguments resolve when the stream ends.
- Latency is measured **first chunk → last chunk** (time-to-first-token visible separately in trace if needed).

### LLMCallTracer
Context manager wrapping every `stream_complete()` call. On stream start: records timestamp + start timer. On each content chunk: invokes `token_callback`. On final chunk: records prompt tokens, completion tokens (from `usage` chunk), total latency_ms, full content. Appends `TraceStep` to `memory.trace`, increments `memory.step_count`. Invokes async `emit_callback(TraceStep)` for Chainlit step finalisation.

### AzureLLMProvider
- Single `AzureOpenAI` client instantiated once
- All calls use `stream=True`, `stream_options={"include_usage": True}`
- Retry up to 3 times on `APITimeoutError` / `APIError` (retry restarts the stream from the beginning)
- On final failure: records `[LLM_CALL_FAILED]` in trace, returns gracefully — never raises to caller
- All retries logged in trace

---

## Component 3: Tool Registry + Class-Based Tools

**Files:** `src/agents/tool_registry.py`, `src/agents/tools/`

### BaseTool (ABC)
```
name: str
description: str
parameters: dict         # JSON schema → OpenAI function definition
execute(inputs, memory) → str
to_openai_definition() → dict
```

### ToolRegistry
Stores `BaseTool` instances by name. Methods:
- `register(tool: BaseTool)`
- `get_openai_definitions(names?) → list[dict]`
- `dispatch(tool_name, inputs, memory) → str` — catches all exceptions, returns `[TOOL_ERROR: ...]` string, never raises

### Tools

| Tool | Constructor deps | Description |
|------|-----------------|-------------|
| `ReadPDFTool` | none | PyMuPDF primary; if page text < `pdf_ocr_fallback_min_chars`, render page to image and run pytesseract. On total failure: `extraction_method="failed"`, appends SourceDocument anyway |
| `ExtractSectionTool` | `BaseLLMProvider` | LLM call to extract a named section. Prompt instructs: return verbatim quote only, never infer. Returns `ExtractedField` with `confidence` and `raw_quote` |
| `ReconcileMedicationsTool` | none | Diffs admission vs discharge medication lists. Flags `RECONCILIATION_NEEDED` for: new med (no reason), stopped med (no reason), dose change (no reason). Writes `ClinicalFlag` entries |
| `DetectConflictsTool` | none | Given `{doc: value}` dict for a field, if values differ → writes `Conflict` + `ClinicalFlag(severity="CONFLICT")` |
| `DrugInteractionLookupTool` | none | Mocked. Checks discharge med list against a hardcoded interaction table (clinically real pairs). Returns warnings. On any exception → `[DRUG_LOOKUP_UNAVAILABLE]`, agent continues |
| `FlagForClinicianReviewTool` | none | Writes `ClinicalFlag` to memory. Always succeeds — the safety valve |
| `ValidateCompletenessTool` | `BaseLLMProvider` | Critic only. LLM call to verify all required discharge summary sections are present and sourced or properly flagged |

### Wiring
```python
build_executor_registry(provider) → ToolRegistry   # all 6 executor tools
build_critic_registry(provider)   → ToolRegistry   # detect_conflicts, validate_completeness, flag_for_clinician_review
```

---

## Component 4: Agent Loop Design

**Files:** `src/agents/base_agent.py`, `src/agents/executor_agent.py`, `src/agents/critic_agent.py`, `src/agents/orchestrator.py`

### Executor Agent

System prompt enforces: extract only what is explicitly stated, mark everything else MISSING/PENDING, call detect_conflicts when sources disagree, call flag_for_clinician_review for all gaps.

Workflow the LLM is instructed to follow:
1. `read_pdf` for every document in the patient folder
2. `extract_section` for each of the 11 required discharge summary fields
3. `reconcile_medications` once all medication data is extracted
4. `drug_interaction_lookup` on the discharge medication list
5. Return with no tool call → compile draft → orchestrator hands off to Critic

Loop: build messages → `complete_with_tool_choice` → if tool_calls dispatch all → append results → repeat. If no tool call → compile draft. If step cap reached → append `MISSING` flag and exit.

**`address_feedback(issues, memory)`** — called by orchestrator after Critic returns issues. Starts a fresh message context (not reusing the original long context) with: current draft + critic issues injected as a focused user message. Keeps token count bounded. The Executor runs a targeted pass to resolve only the flagged items.

### Critic Agent

System prompt: strict safety reviewer. Checks fabrication (value vs raw_quote), required fields not flagged, medication reconciliation gaps, unsurfaced conflicts, drug interactions.

Returns `CriticFeedback(approved: bool, issues: list[str])`. If approved → done. If issues → structured list returned to orchestrator for Executor's next pass.

### Orchestrator (handoff loop)

```
memory = SharedMemory(patient_id)
executor.run(patient_dir, memory)              # round 0

for handoff_round in 1..agent_max_handoff_rounds:
    feedback = critic.review(memory)
    if feedback.approved:
        memory.critic_approved = True; break
    executor.address_feedback(feedback.issues, memory)

if not memory.critic_approved:
    append flag: "Critic did not approve — full clinician review required"
return memory
```

Neither agent imports the other. The orchestrator owns handoff routing.

### Hard limits
- `agent_max_steps = 20` — total LLM calls across both agents and all rounds
- `agent_max_handoff_rounds = 3` — critic↔executor iterations
- Both enforced via `memory.step_count` and `memory.handoff_round` checked at loop entry

---

## Component 5: Chainlit UI

**File:** `chainlit_app.py`

### Flow
1. `on_chat_start` — welcome message, prompt for patient folder path
2. `on_message` — validate path → build provider/agents → run orchestrator asynchronously
3. Each `TraceStep` → `cl.Step` rendered in real time (collapsible, shows reasoning + inputs + result + tokens + latency)
4. Handoff events shown as inline banners
5. On completion: discharge summary → flags panel → run stats

### Real-time step rendering
Two callbacks registered before orchestrator runs:
- `token_callback(chunk: str)` — called per streaming chunk, pipes tokens to the active `cl.Step` via `.stream_token()`. Reasoning text appears live.
- `emit_callback(TraceStep)` — called when a step completes, finalises the `cl.Step` with token count + latency. No polling, no threads.

Tool call steps show the tool name immediately when the stream begins; arguments resolve and the step finalises when the stream ends.

### Output sections
- **Discharge Summary** — all 11 required fields, each with confidence badge (`✓ found`, `⏳ pending`, `⚠ missing`)
- **Clinical Flags Panel** — severity-coloured list (`🔴 CONFLICT`, `⚠️ MISSING`, `⏳ PENDING`, `💊 RECONCILIATION_NEEDED`, `🚨 SAFETY`)
- **Run Stats** — step count, handoff rounds, total tokens, avg latency, critic approval status

---

## Required Discharge Summary Sections (11 fields)

```
patient_demographics
admission_date
discharge_date
principal_diagnosis
secondary_diagnoses
hospital_course
procedures
discharge_medications          # with reconciliation changes clearly noted
allergies
follow_up_instructions
pending_results
discharge_condition
```

---

## No-Fabrication Guardrail — How It Works

Three layers:

1. **Prompt layer:** Executor system prompt instructs the LLM to return only verbatim text from source, never infer. `ExtractSectionTool` prompt repeats this constraint.
2. **Data layer:** `ExtractedField.raw_quote` stores the verbatim source anchor. `confidence` is set by the LLM: `found | missing | pending`. The agent cannot output `found` without a non-null `raw_quote`.
3. **Critic layer:** `ValidateCompletenessTool` cross-checks every `found` field's `value` against its `raw_quote`. Any divergence → `flag_for_clinician_review`. The Critic cannot approve a draft with unsourced `found` fields.

---

## Failure Handling

| Failure | Response |
|---------|----------|
| PDF unreadable (PyMuPDF + OCR both fail) | `SourceDocument(extraction_method="failed")`, field flagged `MISSING` |
| LLM call timeout / API error | Retry ×3, then record `[LLM_CALL_FAILED]` in trace, return gracefully |
| Tool not found in registry | Returns `[TOOL_NOT_FOUND: name]`, agent sees it and can re-plan |
| Tool execution exception | `dispatch()` catches, returns `[TOOL_ERROR: name — reason]` |
| Step cap reached | Append `MISSING` flag, exit loop cleanly |
| Critic does not approve after max handoff rounds | Append flag requiring full clinician review |

The agent never crashes. Every failure is observable in the trace and surfaced in the flags panel.

---

## Patient Data Notes (Patient 2)

Known conflicts in the provided data that the agent must surface:
- **Diagnosis conflict:** Discharge summary lists *Acute Gastroenteritis + UTI*; ER observation chart shows *DKA* (blood glucose 443 mg/dl, BP 87/50) — `🔴 CONFLICT` expected
- **Pending result:** Urine culture and sensitivity sent, report awaited — `⏳ PENDING` expected
- **Medication gaps:** Some discharge medications have no dosage listed (Tab M Strong, Tab Zedott, Tab Entro, Tab Meftal Spas) — `💊 RECONCILIATION_NEEDED` expected
- **Discharge against medical advice:** Documented in hospital course — must appear in `hospital_course` field

---

## Environment & Setup

```bash
conda create -n healthcare python=3.11
conda activate healthcare
pip install -e .

# Install tesseract (Windows)
# Download installer from UB-Mannheim/tesseract on GitHub
# Add to PATH

chainlit run chainlit_app.py
```

**Dependencies (pyproject.toml):**
```
openai>=1.0
pydantic>=2.0
pydantic-settings>=2.0
PyMuPDF
pytesseract
Pillow
chainlit
python-dotenv
```

---

## Out of Scope (Part 1)

- Part 2 learning loop (planned separately)
- Real drug interaction API (mocked)
- Multi-patient batch processing (single patient per run)
- Authentication / role-based access
- Persistent storage of drafts

---

## Open Questions (for implementation)

- Exact OpenAI function-calling schema for each tool's `parameters` dict — defined during implementation
- Async/sync callback bridge: `LLMCallTracer.record()` runs in a sync context but `emit_callback` is an async Chainlit coroutine. Resolution: use `asyncio.get_event_loop().run_until_complete()` or run the orchestrator inside `cl.make_async()` so the event loop is always available. Prefer the latter.
- Synthetic patient 1 data generation — deferred to later in dev, needed before video demo
