# Healthcare Discharge Summary Agent

## Project Context
Take-home assignment for Dscribe. Agentic AI system that reads patient source-note PDFs and produces structured discharge summaries for clinician review.

**Assignment file:** `data/Discharge_Summary_Agent_TakeHome_Task_Dscribe_260603_095110.pdf`
**Patient data:** `data/patient 2 (1)_260603_095051.pdf`

## Scope
- **Part 1 (current):** Discharge Summary Agent — required, the submission bar
- **Part 2 (later):** Learning from Doctor Edits — stretch, plan after Part 1 is complete

## Environment
- **Platform:** Windows 11, conda environment
- **LLM:** Azure OpenAI (credentials in `.env`, never commit)
- **UI:** Chainlit (preferred) or Streamlit for agentic flow demo
- **Python packaging:** `pyproject.toml`

## Azure LLM Config (from .env)
```
AZURE_OPENAI_ENDPOINT=...
AZURE_LLM_DEPLOYMENT=gpt-4o
AZURE_LLM_API_VERSION=2025-01-01-preview
AZURE_OPENAI_API_KEY=...
```
Load via Pydantic `BaseSettings` in `src/config/settings.py`.

## Code Structure
```
project/
    src/
        providers/
            base/
                llm_provider.py      # Abstract base class
            azure/
                llm_provider.py      # Azure OpenAI implementation
        config/
            settings.py              # Pydantic BaseSettings from .env
        agents/                      # Agent loop lives here
    chainlit_app.py                  # Demo UI
    pyproject.toml
    .gitignore
    .dockerignore
    Dockerfile
```

## Architecture Principles
- **Abstract provider pattern:** All LLM calls go through `BaseLLMProvider` interface; swap Azure ↔ OpenAI ↔ other without touching agent code
- **Real agent loop:** Plan → Tool call → Observe → Re-plan (not a hardcoded pipeline)
- **No fabrication guardrail:** Any field not sourced from documents = marked `[MISSING]` or `[PENDING]`, flagged for clinician review
- **Observability:** Each step emits a structured trace: `reasoning → tool chosen → inputs → result → next decision`
- **Hard iteration cap:** Agent cannot loop forever; configurable max steps

## Key Agent Tools (Part 1)
- `read_pdf(path)` — extract text from a patient PDF
- `drug_interaction_lookup(medications)` — mock external tool
- `flag_for_clinician_review(field, reason)` — escalation action
- `reconcile_medications(admission_list, discharge_list)` — surface changes
- `detect_conflicts(field, values_from_sources)` — flag contradictions

## Running the Project
```bash
# Setup
conda create -n healthcare python=3.11
conda activate healthcare
pip install -e .

# Run UI
chainlit run chainlit_app.py

# Run agent headless (for a patient folder)
python -m src.agents.discharge_agent --patient-dir data/patient2/
```

## What to Avoid
- Never commit `.env` or any API keys
- Never upload patient data to third-party services
- Never have the agent fill in a missing clinical fact with a plausible value
- Do not use LangGraph/CrewAI (from-scratch agent loop preferred per assignment)

## Key Documents
- **Design spec:** `docs/superpowers/specs/2026-06-03-discharge-summary-agent-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-06-03-discharge-summary-agent.md`

## Current Status
- [ ] Part 1: Discharge Summary Agent (plan ready, implementation not started)
- [ ] Part 2: Learning from Doctor Edits (not started)
