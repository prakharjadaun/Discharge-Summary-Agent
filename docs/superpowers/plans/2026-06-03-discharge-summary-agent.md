# Discharge Summary Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-agent (Executor + Critic) agentic system that reads patient PDFs and produces a clinically safe discharge summary draft, with real-time streaming output in a Chainlit UI.

**Architecture:** Executor agent ingests PDFs, extracts all clinical sections using tools, then hands off to a Critic agent that checks for fabrication, conflicts, and missing fields. Both share a `SharedMemory` Pydantic model. All LLM calls go through `AsyncAzureLLMProvider`, which streams tokens to Chainlit and records every call in a `TraceStep`.

**Tech Stack:** Python 3.11, conda, Azure OpenAI (gpt-4o), PyMuPDF, pytesseract, Chainlit, Pydantic v2, pydantic-settings, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-06-03-discharge-summary-agent-design.md`

---

## File Map

```
src/
  config/
    __init__.py
    settings.py                        # Pydantic BaseSettings from .env
  providers/
    __init__.py
    base/
      __init__.py
      llm_provider.py                  # BaseLLMProvider ABC (async)
      tracer.py                        # LLMCallTracer context manager
    azure/
      __init__.py
      llm_provider.py                  # AsyncAzureLLMProvider (streaming)
  agents/
    __init__.py
    shared_memory.py                   # All data models + SharedMemory
    base_agent.py                      # BaseAgent with shared loop utilities
    executor_agent.py                  # ExecutorAgent
    critic_agent.py                    # CriticAgent + CriticFeedback
    orchestrator.py                    # DischargeAgentOrchestrator
    tool_registry.py                   # ToolRegistry
    tools/
      __init__.py                      # build_executor_registry(), build_critic_registry()
      base.py                          # BaseTool ABC
      read_pdf.py                      # ReadPDFTool
      extract_section.py               # ExtractSectionTool (needs LLM)
      reconcile_medications.py         # ReconcileMedicationsTool
      detect_conflicts.py              # DetectConflictsTool
      drug_interaction_lookup.py       # DrugInteractionLookupTool (mocked)
      flag_for_clinician_review.py     # FlagForClinicianReviewTool
      validate_completeness.py         # ValidateCompletenessTool (needs LLM)
tests/
  __init__.py
  test_settings.py
  test_shared_memory.py
  test_tool_registry.py
  test_tracer.py
  test_azure_provider.py
  tools/
    __init__.py
    test_read_pdf.py
    test_extract_section.py
    test_reconcile_medications.py
    test_detect_conflicts.py
    test_drug_interaction_lookup.py
    test_flag_for_clinician_review.py
    test_validate_completeness.py
  agents/
    __init__.py
    test_executor_agent.py
    test_critic_agent.py
    test_orchestrator.py
chainlit_app.py
pyproject.toml
.env                                   # never committed
.gitignore
.dockerignore
Dockerfile
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.dockerignore`
- Create: `Dockerfile`
- Create: all `__init__.py` stubs

- [ ] **Step 1: Create conda environment**

```bash
conda create -n healthcare python=3.11 -y
conda activate healthcare
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "healthcare-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.55",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "PyMuPDF>=1.24",
    "pytesseract>=0.3",
    "Pillow>=10.0",
    "chainlit>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: all packages install without errors.

- [ ] **Step 4: Install tesseract (Windows)**

Download the UB-Mannheim tesseract installer from:
`https://github.com/UB-Mannheim/tesseract/wiki`
Install to `C:\Program Files\Tesseract-OCR\` and add to PATH.

Verify:
```bash
tesseract --version
```

- [ ] **Step 5: Create directory structure and `__init__.py` stubs**

```bash
mkdir -p src/config src/providers/base src/providers/azure
mkdir -p src/agents/tools
mkdir -p tests/tools tests/agents

touch src/__init__.py src/config/__init__.py
touch src/providers/__init__.py src/providers/base/__init__.py src/providers/azure/__init__.py
touch src/agents/__init__.py src/agents/tools/__init__.py
touch tests/__init__.py tests/tools/__init__.py tests/agents/__init__.py
```

- [ ] **Step 6: Write `.gitignore`**

```
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
.chainlit/
```

- [ ] **Step 7: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y tesseract-ocr libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY src/ src/
COPY chainlit_app.py .

EXPOSE 8000
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml .gitignore .dockerignore Dockerfile src/ tests/
git commit -m "feat: project scaffold — structure, deps, dockerfile"
```

---

## Task 2: Config / Settings

**Files:**
- Create: `src/config/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_settings.py
import os
import pytest
from unittest.mock import patch

def test_settings_loads_from_env():
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://example.azure.com/",
        "AZURE_LLM_DEPLOYMENT": "gpt-4o",
        "AZURE_LLM_API_VERSION": "2025-01-01-preview",
        "AZURE_OPENAI_API_KEY": "test-key-123",
    }
    with patch.dict(os.environ, env, clear=True):
        from importlib import reload
        import src.config.settings as s
        reload(s)
        assert s.settings.azure_llm_deployment == "gpt-4o"
        assert s.settings.agent_max_steps == 20
        assert s.settings.agent_max_handoff_rounds == 3
        assert s.settings.pdf_ocr_fallback_min_chars == 50
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_settings.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.config.settings'`

- [ ] **Step 3: Write `src/config/settings.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_llm_deployment: str
    azure_llm_api_version: str
    azure_openai_api_key: str

    agent_max_steps: int = 20
    agent_max_handoff_rounds: int = 3
    pdf_ocr_fallback_min_chars: int = 50

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_settings.py -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/config/settings.py tests/test_settings.py
git commit -m "feat: pydantic settings from .env"
```

---

## Task 3: Shared Memory + Data Models

**Files:**
- Create: `src/agents/shared_memory.py`
- Create: `tests/test_shared_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_shared_memory.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_shared_memory.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/agents/shared_memory.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_shared_memory.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/agents/shared_memory.py tests/test_shared_memory.py
git commit -m "feat: shared memory and clinical data models"
```

---

## Task 4: BaseTool ABC + ToolRegistry

**Files:**
- Create: `src/agents/tools/base.py`
- Create: `src/agents/tool_registry.py`
- Create: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tool_registry.py
import pytest
from src.agents.tools.base import BaseTool
from src.agents.tool_registry import ToolRegistry
from src.agents.shared_memory import SharedMemory

class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the input message"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"]
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        return f"ECHO: {inputs['message']}"

class CrashTool(BaseTool):
    name = "crash"
    description = "Always raises"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        raise ValueError("intentional crash")

@pytest.mark.asyncio
async def test_registry_dispatch_success():
    registry = ToolRegistry()
    registry.register(EchoTool())
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("echo", {"message": "hello"}, mem)
    assert result == "ECHO: hello"

@pytest.mark.asyncio
async def test_registry_dispatch_not_found():
    registry = ToolRegistry()
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("nonexistent", {}, mem)
    assert "[TOOL_NOT_FOUND: nonexistent]" in result

@pytest.mark.asyncio
async def test_registry_dispatch_handles_exception():
    registry = ToolRegistry()
    registry.register(CrashTool())
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("crash", {}, mem)
    assert "[TOOL_ERROR: crash" in result

def test_tool_openai_definition():
    tool = EchoTool()
    defn = tool.to_openai_definition()
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "echo"
    assert "message" in defn["function"]["parameters"]["properties"]

def test_registry_get_definitions():
    registry = ToolRegistry()
    registry.register(EchoTool())
    defs = registry.get_openai_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "echo"

def test_registry_get_definitions_filtered():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CrashTool())
    defs = registry.get_openai_definitions(names=["echo"])
    assert len(defs) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_tool_registry.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/agents/tools/base.py`**

```python
from abc import ABC, abstractmethod
from src.agents.shared_memory import SharedMemory

class BaseTool(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        ...

    def to_openai_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

- [ ] **Step 4: Write `src/agents/tool_registry.py`**

```python
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_openai_definitions(self, names: list[str] | None = None) -> list[dict]:
        tools = self._tools if names is None else {
            k: v for k, v in self._tools.items() if k in names
        }
        return [t.to_openai_definition() for t in tools.values()]

    async def dispatch(self, tool_name: str, inputs: dict, memory: SharedMemory) -> str:
        tool = self._tools.get(tool_name)
        if not tool:
            return f"[TOOL_NOT_FOUND: {tool_name}]"
        try:
            return await tool.execute(inputs, memory)
        except Exception as e:
            return f"[TOOL_ERROR: {tool_name} — {str(e)}]"
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_tool_registry.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/base.py src/agents/tool_registry.py tests/test_tool_registry.py
git commit -m "feat: BaseTool ABC and ToolRegistry with async dispatch"
```

---

## Task 5: LLMCallTracer

**Files:**
- Create: `src/providers/base/tracer.py`
- Create: `tests/test_tracer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tracer.py
import pytest
import time
from unittest.mock import MagicMock, AsyncMock
from src.providers.base.tracer import LLMCallTracer
from src.agents.shared_memory import SharedMemory

def make_mock_response(content="test output", prompt_tokens=100, completion_tokens=50):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response

@pytest.mark.asyncio
async def test_tracer_records_step():
    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(
        memory=mem,
        agent="executor",
        action="read_pdf",
        inputs={"path": "file.pdf"}
    )
    tracer.__enter__()
    response = make_mock_response("some content", 100, 50)
    step = await tracer.record("reasoning text", response)

    assert mem.step_count == 1
    assert len(mem.trace) == 1
    assert step.agent == "executor"
    assert step.action == "read_pdf"
    assert step.reasoning == "reasoning text"
    assert step.tokens_in == 100
    assert step.tokens_out == 50
    assert step.latency_ms > 0

@pytest.mark.asyncio
async def test_tracer_record_failure():
    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(memory=mem, agent="executor", action="plan", inputs={})
    tracer.__enter__()
    step = await tracer.record_failure("timeout after 30s")

    assert "[LLM_CALL_FAILED]" in step.result
    assert mem.step_count == 1

@pytest.mark.asyncio
async def test_tracer_calls_emit_callback():
    emitted = []
    async def fake_callback(step):
        emitted.append(step)

    mem = SharedMemory(patient_id="p001")
    tracer = LLMCallTracer(
        memory=mem, agent="executor", action="plan",
        inputs={}, emit_callback=fake_callback
    )
    tracer.__enter__()
    await tracer.record("reasoning", make_mock_response())
    assert len(emitted) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_tracer.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/providers/base/tracer.py`**

```python
from __future__ import annotations
import time
from datetime import datetime
from typing import Awaitable, Callable
from src.agents.shared_memory import SharedMemory, TraceStep

class LLMCallTracer:
    def __init__(
        self,
        memory: SharedMemory,
        agent: str,
        action: str,
        inputs: dict,
        emit_callback: Callable[[TraceStep], Awaitable[None]] | None = None,
    ):
        self._memory = memory
        self._agent = agent
        self._action = action
        self._inputs = inputs
        self._emit_callback = emit_callback
        self._start: float | None = None

    def __enter__(self) -> "LLMCallTracer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        pass

    async def record(self, reasoning: str, response) -> TraceStep:
        latency_ms = round((time.perf_counter() - self._start) * 1000, 2)
        step = TraceStep(
            step_id=self._memory.step_count,
            agent=self._agent,
            reasoning=reasoning,
            action=self._action,
            inputs=self._inputs,
            result=response.choices[0].message.content or "",
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )
        self._memory.trace.append(step)
        self._memory.step_count += 1
        if self._emit_callback:
            await self._emit_callback(step)
        return step

    async def record_failure(self, error: str) -> TraceStep:
        latency_ms = round((time.perf_counter() - self._start) * 1000, 2)
        step = TraceStep(
            step_id=self._memory.step_count,
            agent=self._agent,
            reasoning="LLM call failed",
            action=self._action,
            inputs=self._inputs,
            result=f"[LLM_CALL_FAILED: {error}]",
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
        )
        self._memory.trace.append(step)
        self._memory.step_count += 1
        if self._emit_callback:
            await self._emit_callback(step)
        return step
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_tracer.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/providers/base/tracer.py tests/test_tracer.py
git commit -m "feat: LLMCallTracer — records tokens, latency, emit callback"
```

---

## Task 6: BaseLLMProvider + AsyncAzureLLMProvider (Streaming)

**Files:**
- Create: `src/providers/base/llm_provider.py`
- Create: `src/providers/azure/llm_provider.py`
- Create: `tests/test_azure_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_azure_provider.py
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from src.agents.shared_memory import SharedMemory

def make_stream_chunks(content="hello world", tool_name=None, tool_args=None):
    """Build a list of mock async stream chunks."""
    chunks = []
    if content:
        for char in content:
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = char
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)

    if tool_name:
        tc_chunk = MagicMock()
        tc_chunk.usage = None
        tc_chunk.choices = [MagicMock()]
        tc_chunk.choices[0].delta.content = None
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_123"
        tc.function.name = tool_name
        tc.function.arguments = tool_args or "{}"
        tc_chunk.choices[0].delta.tool_calls = [tc]
        chunks.append(tc_chunk)

    usage_chunk = MagicMock()
    usage_chunk.usage = MagicMock()
    usage_chunk.usage.prompt_tokens = 100
    usage_chunk.usage.completion_tokens = 50
    usage_chunk.choices = []
    chunks.append(usage_chunk)
    return chunks

async def async_iter(items):
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_stream_complete_text_response():
    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI") as MockClient:
        instance = MockClient.return_value
        chunks = make_stream_chunks(content="test response")
        instance.chat.completions.create = AsyncMock(
            return_value=async_iter(chunks)
        )

        from src.providers.azure.llm_provider import AsyncAzureLLMProvider
        mem = SharedMemory(patient_id="test")

        with patch("src.providers.azure.llm_provider.settings") as mock_settings:
            mock_settings.azure_openai_endpoint = "https://test.azure.com/"
            mock_settings.azure_openai_api_key = "key"
            mock_settings.azure_llm_api_version = "2025-01-01-preview"
            mock_settings.azure_llm_deployment = "gpt-4o"
            provider = AsyncAzureLLMProvider()
            provider._client = instance
            provider._deployment = "gpt-4o"

        tokens = []
        async def collect(chunk): tokens.append(chunk)

        content, tool_calls, step = await provider.stream_complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            memory=mem,
            agent="executor",
            action="plan",
            inputs={},
            token_callback=collect,
        )

        assert content == "test response"
        assert tool_calls is None
        assert step.tokens_in == 100
        assert step.tokens_out == 50
        assert "".join(tokens) == "test response"

@pytest.mark.asyncio
async def test_stream_complete_tool_call():
    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI") as MockClient:
        instance = MockClient.return_value
        chunks = make_stream_chunks(
            content=None,
            tool_name="read_pdf",
            tool_args='{"path": "test.pdf"}'
        )
        instance.chat.completions.create = AsyncMock(
            return_value=async_iter(chunks)
        )

        from src.providers.azure.llm_provider import AsyncAzureLLMProvider
        mem = SharedMemory(patient_id="test")
        provider = AsyncAzureLLMProvider()
        provider._client = instance
        provider._deployment = "gpt-4o"

        content, tool_calls, step = await provider.stream_complete(
            messages=[{"role": "user", "content": "process files"}],
            tools=[{"type": "function", "function": {"name": "read_pdf"}}],
            memory=mem,
            agent="executor",
            action="plan",
            inputs={},
        )

        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "read_pdf"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "test.pdf"}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_azure_provider.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/providers/base/llm_provider.py`**

```python
from abc import ABC, abstractmethod
from typing import Awaitable, Callable
from src.agents.shared_memory import SharedMemory, TraceStep

class BaseLLMProvider(ABC):

    @abstractmethod
    async def stream_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        memory: SharedMemory,
        agent: str,
        action: str,
        inputs: dict,
        token_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[dict] | None, TraceStep]:
        """
        Returns (content, tool_calls, trace_step).
        tool_calls is None when the model returns a text response.
        token_callback is called per streaming content chunk.
        """
        ...
```

- [ ] **Step 4: Write `src/providers/azure/llm_provider.py`**

```python
from __future__ import annotations
import json
from typing import Awaitable, Callable
from openai import AsyncAzureOpenAI, APITimeoutError, APIError
from src.providers.base.llm_provider import BaseLLMProvider
from src.providers.base.tracer import LLMCallTracer
from src.agents.shared_memory import SharedMemory, TraceStep
from src.config.settings import settings

class AsyncAzureLLMProvider(BaseLLMProvider):

    def __init__(
        self,
        emit_callback: Callable[[TraceStep], Awaitable[None]] | None = None,
    ):
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_llm_api_version,
        )
        self._deployment = settings.azure_llm_deployment
        self._emit_callback = emit_callback

    async def stream_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        memory: SharedMemory,
        agent: str,
        action: str,
        inputs: dict,
        token_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[dict] | None, TraceStep]:

        with LLMCallTracer(memory, agent, action, inputs, self._emit_callback) as tracer:
            for attempt in range(3):
                try:
                    return await self._do_stream(
                        messages, tools, tracer, token_callback
                    )
                except (APITimeoutError, APIError) as e:
                    if attempt == 2:
                        step = await tracer.record_failure(str(e))
                        return "[LLM_CALL_FAILED]", None, step

    async def _do_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tracer: LLMCallTracer,
        token_callback,
    ) -> tuple[str, list[dict] | None, TraceStep]:

        kwargs = dict(
            model=self._deployment,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        usage = None

        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
                continue
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                if token_callback:
                    await token_callback(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        content = "".join(content_parts)
        tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None

        import types
        fake_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )],
            usage=types.SimpleNamespace(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        )
        reasoning = (
            content[:200] if content
            else f"tool_call: {tool_calls_acc[0]['function']['name']}" if tool_calls_acc
            else "no content"
        )
        step = await tracer.record(reasoning=reasoning, response=fake_response)
        return content, tool_calls, step
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_azure_provider.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/providers/ tests/test_azure_provider.py
git commit -m "feat: async Azure OpenAI provider with streaming and tracer"
```

---

## Task 7: ReadPDFTool

**Files:**
- Create: `src/agents/tools/read_pdf.py`
- Create: `tests/tools/test_read_pdf.py`
- Create: `tests/fixtures/` (small test PDF)

- [ ] **Step 1: Create a minimal test PDF fixture**

```python
# Run once to create the fixture (paste into Python REPL):
import fitz, os
os.makedirs("tests/fixtures", exist_ok=True)
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 100), "DIAGNOSIS: ACUTE GASTROENTERITIS\nPATIENT: Jane Doe")
doc.save("tests/fixtures/sample.pdf")
doc.close()
```

- [ ] **Step 2: Write failing tests**

```python
# tests/tools/test_read_pdf.py
import pytest
from pathlib import Path
from src.agents.tools.read_pdf import ReadPDFTool
from src.agents.shared_memory import SharedMemory

FIXTURE_PDF = Path("tests/fixtures/sample.pdf")

@pytest.mark.asyncio
async def test_read_pdf_extracts_text():
    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": str(FIXTURE_PDF)}, mem)

    assert "[READ_PDF_OK]" in result
    assert len(mem.source_documents) == 1
    assert "GASTROENTERITIS" in mem.source_documents[0].raw_text
    assert mem.source_documents[0].extraction_method in ("pymupdf", "ocr")

@pytest.mark.asyncio
async def test_read_pdf_nonexistent_file():
    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": "nonexistent.pdf"}, mem)

    assert "[READ_PDF_FAILED]" in result
    assert len(mem.source_documents) == 1
    assert mem.source_documents[0].extraction_method == "failed"

@pytest.mark.asyncio
async def test_read_pdf_ocr_fallback_triggered_for_sparse_page(tmp_path):
    # Create a PDF with a blank page (no text) to trigger OCR path
    import fitz
    doc = fitz.open()
    doc.new_page()  # blank page
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    tool = ReadPDFTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"path": str(pdf_path)}, mem)

    # Should not crash; extraction_method is "ocr" or "failed" depending on tesseract
    assert mem.source_documents[0].extraction_method in ("ocr", "failed")
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/tools/test_read_pdf.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 4: Write `src/agents/tools/read_pdf.py`**

```python
from __future__ import annotations
import fitz
from PIL import Image
import pytesseract
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, SourceDocument
from src.config.settings import settings

class ReadPDFTool(BaseTool):
    name = "read_pdf"
    description = (
        "Extract text from a patient PDF file. "
        "Uses OCR fallback for scanned or handwritten pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the PDF file"}
        },
        "required": ["path"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        path = inputs["path"]
        try:
            doc = fitz.open(path)
        except Exception as e:
            src = SourceDocument(path=path, page_count=0, raw_text="", extraction_method="failed")
            memory.source_documents.append(src)
            return f"[READ_PDF_FAILED: {path} — {str(e)}]"

        pages_text: list[str] = []
        final_method = "pymupdf"

        for page in doc:
            text = page.get_text().strip()
            if len(text) < settings.pdf_ocr_fallback_min_chars:
                text = self._ocr_page(page)
                if text:
                    final_method = "ocr"
                else:
                    final_method = "failed"
            pages_text.append(text)

        doc.close()
        full_text = "\n".join(pages_text)
        src = SourceDocument(
            path=path,
            page_count=len(pages_text),
            raw_text=full_text,
            extraction_method=final_method,
        )
        memory.source_documents.append(src)
        return f"[READ_PDF_OK] {path} — {len(full_text)} chars via {final_method}"

    def _ocr_page(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/tools/test_read_pdf.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/read_pdf.py tests/tools/test_read_pdf.py tests/fixtures/
git commit -m "feat: ReadPDFTool with PyMuPDF and tesseract OCR fallback"
```

---

## Task 8: ExtractSectionTool

**Files:**
- Create: `src/agents/tools/extract_section.py`
- Create: `tests/tools/test_extract_section.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_extract_section.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.tools.extract_section import ExtractSectionTool
from src.agents.shared_memory import SharedMemory, SourceDocument, TraceStep
from datetime import datetime

def make_provider(content: str):
    provider = MagicMock()
    step = TraceStep(
        step_id=0, agent="executor", reasoning=content, action="extract_section",
        inputs={}, result=content, tokens_in=50, tokens_out=30,
        latency_ms=200.0, timestamp=datetime.utcnow()
    )
    provider.stream_complete = AsyncMock(return_value=(content, None, step))
    return provider

def make_memory_with_doc(text: str) -> SharedMemory:
    mem = SharedMemory(patient_id="p001")
    mem.source_documents.append(SourceDocument(
        path="discharge.pdf", page_count=1,
        raw_text=text, extraction_method="pymupdf"
    ))
    return mem

@pytest.mark.asyncio
async def test_extract_section_found():
    llm_response = '{"confidence": "found", "value": "Acute Gastroenteritis", "raw_quote": "DIAGNOSIS: ACUTE GASTROENTERITIS"}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("DIAGNOSIS: ACUTE GASTROENTERITIS\nUTI")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "principal_diagnosis"}, mem
    )

    assert "[EXTRACTED]" in result
    assert "principal_diagnosis" in mem.extracted_sections
    field = mem.extracted_sections["principal_diagnosis"]
    assert field.confidence == "found"
    assert field.value == "Acute Gastroenteritis"
    assert field.raw_quote == "DIAGNOSIS: ACUTE GASTROENTERITIS"

@pytest.mark.asyncio
async def test_extract_section_missing():
    llm_response = '{"confidence": "missing", "value": null, "raw_quote": null}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("Some other content with no allergies listed")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "allergies"}, mem
    )

    assert "[MISSING]" in result
    field = mem.extracted_sections["allergies"]
    assert field.confidence == "missing"
    assert field.value is None
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "MISSING"

@pytest.mark.asyncio
async def test_extract_section_pending():
    llm_response = '{"confidence": "pending", "value": "Urine culture — report awaited", "raw_quote": "urine culture and sensitivity sent- report awaited"}'
    provider = make_provider(llm_response)
    tool = ExtractSectionTool(provider)
    mem = make_memory_with_doc("urine culture and sensitivity sent- report awaited")

    result = await tool.execute(
        {"document_path": "discharge.pdf", "section_name": "pending_results"}, mem
    )

    assert "[PENDING]" in result
    field = mem.extracted_sections["pending_results"]
    assert field.confidence == "pending"
    assert len(mem.flags) == 1
    assert mem.flags[0].severity == "PENDING"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/tools/test_extract_section.py -v
```

- [ ] **Step 3: Write `src/agents/tools/extract_section.py`**

```python
from __future__ import annotations
import json
from src.agents.tools.base import BaseTool
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.shared_memory import SharedMemory, ExtractedField, ClinicalFlag

EXTRACT_SYSTEM_PROMPT = """You are a clinical data extraction assistant.
Extract ONLY text that appears VERBATIM in the source document.
Never infer, assume, or fill gaps.

Return ONLY valid JSON:
{"confidence": "found"|"missing"|"pending", "value": <string or null>, "raw_quote": <verbatim snippet or null>}

- "found": section clearly stated in document. Include the exact quote.
- "pending": explicitly noted as pending/awaited.
- "missing": not present in the document at all.
"""

class ExtractSectionTool(BaseTool):
    name = "extract_section"
    description = "Extract a named clinical section from a source document. Never infers — returns missing if absent."
    parameters = {
        "type": "object",
        "properties": {
            "document_path": {"type": "string"},
            "section_name": {"type": "string"},
        },
        "required": ["document_path", "section_name"],
    }

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        doc_path = inputs["document_path"]
        section_name = inputs["section_name"]

        source = next((d for d in memory.source_documents if d.path == doc_path), None)
        if not source:
            return f"[EXTRACT_ERROR: document {doc_path} not loaded — call read_pdf first]"

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Document text:\n{source.raw_text[:6000]}\n\n"
                f"Extract section: {section_name}"
            )},
        ]

        content, _, _ = await self._provider.stream_complete(
            messages=messages, tools=None,
            memory=memory, agent="executor",
            action="extract_section",
            inputs={"section": section_name, "doc": doc_path},
        )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"confidence": "missing", "value": None, "raw_quote": None}

        confidence = parsed.get("confidence", "missing")
        value = parsed.get("value")
        raw_quote = parsed.get("raw_quote")

        field = ExtractedField(
            value=value, source_doc=doc_path,
            confidence=confidence, raw_quote=raw_quote
        )
        memory.extracted_sections[section_name] = field

        if confidence == "missing":
            memory.flags.append(ClinicalFlag(
                field=section_name,
                reason=f"Section '{section_name}' not found in {doc_path}",
                severity="MISSING",
                source_docs=[doc_path],
            ))
            return f"[MISSING] {section_name} not found in {doc_path}"

        if confidence == "pending":
            memory.flags.append(ClinicalFlag(
                field=section_name,
                reason=f"Section '{section_name}' is pending: {value}",
                severity="PENDING",
                source_docs=[doc_path],
            ))
            return f"[PENDING] {section_name}: {value}"

        return f"[EXTRACTED] {section_name}: {value}"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/tools/test_extract_section.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/extract_section.py tests/tools/test_extract_section.py
git commit -m "feat: ExtractSectionTool with JSON response and anti-fabrication anchor"
```

---

## Task 9: ReconcileMedicationsTool + DetectConflictsTool

**Files:**
- Create: `src/agents/tools/reconcile_medications.py`
- Create: `src/agents/tools/detect_conflicts.py`
- Create: `tests/tools/test_reconcile_medications.py`
- Create: `tests/tools/test_detect_conflicts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_reconcile_medications.py
import pytest
from src.agents.tools.reconcile_medications import ReconcileMedicationsTool
from src.agents.shared_memory import SharedMemory, Medication

@pytest.mark.asyncio
async def test_new_medication_no_reason_flagged():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    mem.admission_medications = [
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False)
    ]
    mem.discharge_medications = [
        Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False),
        Medication(name="TAB. EMESET", dose="4MG", frequency="1-1-1", reason_documented=False),  # NEW
    ]
    result = await tool.execute({}, mem)
    assert "RECONCILIATION_NEEDED" in result
    flags = [f for f in mem.flags if f.severity == "RECONCILIATION_NEEDED"]
    assert len(flags) == 1
    assert "EMESET" in flags[0].reason

@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_continued_medication_no_flag():
    tool = ReconcileMedicationsTool()
    mem = SharedMemory(patient_id="p001")
    med = Medication(name="TAB. RACIPER", dose="40MG", frequency="1-0-0", reason_documented=False)
    mem.admission_medications = [med]
    mem.discharge_medications = [med]
    await tool.execute({}, mem)
    assert len(mem.flags) == 0
```

```python
# tests/tools/test_detect_conflicts.py
import pytest
from src.agents.tools.detect_conflicts import DetectConflictsTool
from src.agents.shared_memory import SharedMemory

@pytest.mark.asyncio
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

@pytest.mark.asyncio
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/tools/test_reconcile_medications.py tests/tools/test_detect_conflicts.py -v
```

- [ ] **Step 3: Write `src/agents/tools/reconcile_medications.py`**

```python
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag

class ReconcileMedicationsTool(BaseTool):
    name = "reconcile_medications"
    description = "Compare admission vs discharge medications and flag undocumented changes."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        admission_names = {m.name.upper(): m for m in memory.admission_medications}
        discharge_names = {m.name.upper(): m for m in memory.discharge_medications}
        findings: list[str] = []

        for name, med in discharge_names.items():
            if name not in admission_names:
                memory.flags.append(ClinicalFlag(
                    field=f"medication:{med.name}",
                    reason=f"NEW medication '{med.name}' on discharge — no documented reason",
                    severity="RECONCILIATION_NEEDED",
                    source_docs=[],
                ))
                findings.append(f"NEW:{med.name}")

        for name, med in admission_names.items():
            if name not in discharge_names:
                memory.flags.append(ClinicalFlag(
                    field=f"medication:{med.name}",
                    reason=f"STOPPED medication '{med.name}' — not on discharge list, no documented reason",
                    severity="RECONCILIATION_NEEDED",
                    source_docs=[],
                ))
                findings.append(f"STOPPED:{med.name}")

        if not findings:
            return "[RECONCILE_OK] No undocumented medication changes found"
        return f"[RECONCILIATION_NEEDED] {'; '.join(findings)}"
```

- [ ] **Step 4: Write `src/agents/tools/detect_conflicts.py`**

```python
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, Conflict, ClinicalFlag

class DetectConflictsTool(BaseTool):
    name = "detect_conflicts"
    description = "Flag when two source documents disagree on the same clinical field."
    parameters = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "values": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map of document_name → value_from_that_doc"
            },
        },
        "required": ["field", "values"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        field = inputs["field"]
        values: dict[str, str] = inputs["values"]

        unique_vals = set(v.strip().lower() for v in values.values())
        if len(unique_vals) <= 1:
            return f"[NO_CONFLICT] {field}: all sources agree"

        conflict = Conflict(field=field, values=values)
        memory.conflicts.append(conflict)
        memory.flags.append(ClinicalFlag(
            field=field,
            reason=f"Conflict in '{field}': {values}",
            severity="CONFLICT",
            source_docs=list(values.keys()),
        ))
        return f"[CONFLICT] {field}: {values}"
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/tools/test_reconcile_medications.py tests/tools/test_detect_conflicts.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/agents/tools/reconcile_medications.py src/agents/tools/detect_conflicts.py \
        tests/tools/test_reconcile_medications.py tests/tools/test_detect_conflicts.py
git commit -m "feat: ReconcileMedicationsTool and DetectConflictsTool"
```

---

## Task 10: DrugInteractionLookupTool + FlagForClinicianReviewTool + ValidateCompletenessTool

**Files:**
- Create: `src/agents/tools/drug_interaction_lookup.py`
- Create: `src/agents/tools/flag_for_clinician_review.py`
- Create: `src/agents/tools/validate_completeness.py`
- Create: `tests/tools/test_drug_interaction_lookup.py`
- Create: `tests/tools/test_flag_for_clinician_review.py`
- Create: `tests/tools/test_validate_completeness.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_drug_interaction_lookup.py
import pytest
from src.agents.tools.drug_interaction_lookup import DrugInteractionLookupTool
from src.agents.shared_memory import SharedMemory

@pytest.mark.asyncio
async def test_known_interaction_flagged():
    tool = DrugInteractionLookupTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"medications": ["warfarin", "aspirin"]}, mem)
    assert "bleeding risk" in result.lower()
    safety_flags = [f for f in mem.flags if f.severity == "SAFETY"]
    assert len(safety_flags) == 1

@pytest.mark.asyncio
async def test_no_interaction():
    tool = DrugInteractionLookupTool()
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({"medications": ["paracetamol", "vitamin_c"]}, mem)
    assert "no known interactions" in result.lower()
    assert len(mem.flags) == 0
```

```python
# tests/tools/test_flag_for_clinician_review.py
import pytest
from src.agents.tools.flag_for_clinician_review import FlagForClinicianReviewTool
from src.agents.shared_memory import SharedMemory

@pytest.mark.asyncio
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
```

```python
# tests/tools/test_validate_completeness.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.tools.validate_completeness import ValidateCompletenessTool
from src.agents.shared_memory import SharedMemory, ExtractedField, TraceStep
from datetime import datetime

def make_provider(content: str):
    provider = MagicMock()
    step = TraceStep(
        step_id=0, agent="critic", reasoning=content, action="validate",
        inputs={}, result=content, tokens_in=50, tokens_out=30,
        latency_ms=100.0, timestamp=datetime.utcnow()
    )
    provider.stream_complete = AsyncMock(return_value=(content, None, step))
    return provider

@pytest.mark.asyncio
async def test_validate_returns_issues_when_fields_missing():
    provider = make_provider('["principal_diagnosis is marked found but raw_quote is null"]')
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    mem.extracted_sections["principal_diagnosis"] = ExtractedField(
        value="Gastroenteritis", source_doc="note.pdf",
        confidence="found", raw_quote=None  # missing quote = fabrication signal
    )
    result = await tool.execute({}, mem)
    assert "principal_diagnosis" in result

@pytest.mark.asyncio
async def test_validate_returns_ok_when_all_clear():
    provider = make_provider('[]')
    tool = ValidateCompletenessTool(provider)
    mem = SharedMemory(patient_id="p001")
    result = await tool.execute({}, mem)
    assert "[VALIDATION_OK]" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/tools/test_drug_interaction_lookup.py tests/tools/test_flag_for_clinician_review.py tests/tools/test_validate_completeness.py -v
```

- [ ] **Step 3: Write `src/agents/tools/drug_interaction_lookup.py`**

```python
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag

KNOWN_INTERACTIONS: dict[frozenset, str] = {
    frozenset(["warfarin", "aspirin"]): "Increased bleeding risk",
    frozenset(["warfarin", "ibuprofen"]): "Increased bleeding risk",
    frozenset(["metformin", "contrast_dye"]): "Lactic acidosis risk",
    frozenset(["ssri", "tramadol"]): "Serotonin syndrome risk",
    frozenset(["ace_inhibitor", "potassium"]): "Hyperkalaemia risk",
}

class DrugInteractionLookupTool(BaseTool):
    name = "drug_interaction_lookup"
    description = "Check discharge medications for known drug interactions. Returns warnings if found."
    parameters = {
        "type": "object",
        "properties": {
            "medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of medication names (lowercase)"
            }
        },
        "required": ["medications"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        meds = [m.lower().strip() for m in inputs.get("medications", [])]
        warnings: list[str] = []

        for pair, warning in KNOWN_INTERACTIONS.items():
            if pair.issubset(set(meds)):
                warnings.append(f"{' + '.join(pair)}: {warning}")
                memory.flags.append(ClinicalFlag(
                    field="drug_interaction",
                    reason=warning,
                    severity="SAFETY",
                    source_docs=[],
                ))

        if not warnings:
            return "[DRUG_LOOKUP_OK] No known interactions found"
        return f"[DRUG_INTERACTION_WARNING] {'; '.join(warnings)}"
```

- [ ] **Step 4: Write `src/agents/tools/flag_for_clinician_review.py`**

```python
from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory, ClinicalFlag

class FlagForClinicianReviewTool(BaseTool):
    name = "flag_for_clinician_review"
    description = "Escalate a field or finding for clinician review. Always succeeds."
    parameters = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "reason": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["MISSING", "PENDING", "CONFLICT", "RECONCILIATION_NEEDED", "SAFETY"]
            },
        },
        "required": ["field", "reason", "severity"],
    }

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        flag = ClinicalFlag(
            field=inputs["field"],
            reason=inputs["reason"],
            severity=inputs["severity"],
            source_docs=[],
        )
        memory.flags.append(flag)
        return f"[FLAGGED] {inputs['severity']} — {inputs['field']}: {inputs['reason']}"
```

- [ ] **Step 5: Write `src/agents/tools/validate_completeness.py`**

```python
from __future__ import annotations
import json
from src.agents.tools.base import BaseTool
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.shared_memory import SharedMemory

REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]

VALIDATE_SYSTEM_PROMPT = """You are a clinical safety reviewer checking a discharge summary draft.
Check ONLY for these issues:
1. A field has confidence="found" but raw_quote is null (fabrication risk)
2. A required section is completely absent from extracted_sections

Return a JSON array of issue strings. Return [] if everything is clean.
Example: ["principal_diagnosis is marked found but raw_quote is null", "allergies section missing"]
"""

class ValidateCompletenessTool(BaseTool):
    name = "validate_completeness"
    description = "Verify all required discharge summary sections are present and sourced."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def execute(self, inputs: dict, memory: SharedMemory) -> str:
        sections_summary = {
            name: {
                "confidence": field.confidence,
                "has_raw_quote": field.raw_quote is not None,
                "value_present": field.value is not None,
            }
            for name, field in memory.extracted_sections.items()
        }
        missing_sections = [s for s in REQUIRED_SECTIONS if s not in memory.extracted_sections]

        messages = [
            {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Extracted sections: {json.dumps(sections_summary)}\n"
                f"Missing sections: {missing_sections}\n"
                f"Existing flags count: {len(memory.flags)}"
            )},
        ]

        content, _, _ = await self._provider.stream_complete(
            messages=messages, tools=None,
            memory=memory, agent="critic",
            action="validate_completeness", inputs={},
        )

        try:
            issues = json.loads(content)
        except json.JSONDecodeError:
            issues = []

        if not issues:
            return "[VALIDATION_OK] All sections present and sourced"
        return f"[VALIDATION_ISSUES] {'; '.join(issues)}"
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
pytest tests/tools/ -v
```

Expected: all `PASSED`

- [ ] **Step 7: Commit**

```bash
git add src/agents/tools/drug_interaction_lookup.py src/agents/tools/flag_for_clinician_review.py \
        src/agents/tools/validate_completeness.py tests/tools/
git commit -m "feat: DrugInteractionLookupTool, FlagForClinicianReviewTool, ValidateCompletenessTool"
```

---

## Task 11: Tool Registry Wiring

**Files:**
- Modify: `src/agents/tools/__init__.py`

- [ ] **Step 1: Write `src/agents/tools/__init__.py`**

```python
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.tool_registry import ToolRegistry
from src.agents.tools.read_pdf import ReadPDFTool
from src.agents.tools.extract_section import ExtractSectionTool
from src.agents.tools.reconcile_medications import ReconcileMedicationsTool
from src.agents.tools.detect_conflicts import DetectConflictsTool
from src.agents.tools.drug_interaction_lookup import DrugInteractionLookupTool
from src.agents.tools.flag_for_clinician_review import FlagForClinicianReviewTool
from src.agents.tools.validate_completeness import ValidateCompletenessTool

def build_executor_registry(provider: BaseLLMProvider) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadPDFTool())
    registry.register(ExtractSectionTool(provider))
    registry.register(ReconcileMedicationsTool())
    registry.register(DetectConflictsTool())
    registry.register(DrugInteractionLookupTool())
    registry.register(FlagForClinicianReviewTool())
    return registry

def build_critic_registry(provider: BaseLLMProvider) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DetectConflictsTool())
    registry.register(ValidateCompletenessTool(provider))
    registry.register(FlagForClinicianReviewTool())
    return registry
```

- [ ] **Step 2: Verify wiring works**

```bash
python -c "
from unittest.mock import MagicMock
from src.agents.tools import build_executor_registry, build_critic_registry
p = MagicMock()
er = build_executor_registry(p)
cr = build_critic_registry(p)
print('Executor tools:', [d['function']['name'] for d in er.get_openai_definitions()])
print('Critic tools:', [d['function']['name'] for d in cr.get_openai_definitions()])
"
```

Expected output:
```
Executor tools: ['read_pdf', 'extract_section', 'reconcile_medications', 'detect_conflicts', 'drug_interaction_lookup', 'flag_for_clinician_review']
Critic tools: ['detect_conflicts', 'validate_completeness', 'flag_for_clinician_review']
```

- [ ] **Step 3: Commit**

```bash
git add src/agents/tools/__init__.py
git commit -m "feat: tool registry wiring — executor and critic registries"
```

---

## Task 12: BaseAgent + ExecutorAgent

**Files:**
- Create: `src/agents/base_agent.py`
- Create: `src/agents/executor_agent.py`
- Create: `tests/agents/test_executor_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_executor_agent.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from src.agents.executor_agent import ExecutorAgent
from src.agents.shared_memory import SharedMemory, SourceDocument, TraceStep
from datetime import datetime

def make_step():
    return TraceStep(
        step_id=0, agent="executor", reasoning="", action="plan",
        inputs={}, result="", tokens_in=10, tokens_out=5,
        latency_ms=100.0, timestamp=datetime.utcnow()
    )

def make_provider_sequence(responses: list):
    """Returns a provider that cycles through (content, tool_calls) tuples."""
    call_count = 0
    async def stream_complete(*args, **kwargs):
        nonlocal call_count
        content, tool_calls = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return content, tool_calls, make_step()
    provider = MagicMock()
    provider.stream_complete = stream_complete
    return provider

@pytest.mark.asyncio
async def test_executor_compiles_draft_after_no_tool_call(tmp_path):
    # First call returns a tool call (read_pdf), second returns no tool call (done)
    read_tool_call = [{
        "id": "call_1", "type": "function",
        "function": {"name": "read_pdf", "arguments": json.dumps({"path": "test.pdf"})}
    }]

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[READ_PDF_OK] test.pdf — 500 chars via pymupdf")

    provider = make_provider_sequence([
        ("", read_tool_call),   # first call: read_pdf tool call
        ("COMPILE", None),      # second call: done
    ])

    agent = ExecutorAgent(provider, registry, "executor")
    mem = SharedMemory(patient_id="p001")
    # Pre-populate a source document so compile has something
    mem.source_documents.append(SourceDocument(
        path="test.pdf", page_count=1, raw_text="content", extraction_method="pymupdf"
    ))
    await agent.run(patient_dir=str(tmp_path), memory=mem)

    assert mem.draft is not None

@pytest.mark.asyncio
async def test_executor_respects_step_cap():
    # Provider always returns a tool call — agent must stop at cap
    tool_call = [{"id": "c", "type": "function", "function": {"name": "read_pdf", "arguments": "{\"path\": \"f.pdf\"}"}}]

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[READ_PDF_OK]")

    responses = [("", tool_call)] * 25  # more than max_steps
    provider = make_provider_sequence(responses)

    from unittest.mock import patch
    with patch("src.agents.executor_agent.settings") as ms:
        ms.agent_max_steps = 5
        agent = ExecutorAgent(provider, registry, "executor")
        mem = SharedMemory(patient_id="p001")
        await agent.run(patient_dir=".", memory=mem)

    cap_flags = [f for f in mem.flags if "cap" in f.reason.lower()]
    assert len(cap_flags) >= 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agents/test_executor_agent.py -v
```

- [ ] **Step 3: Write `src/agents/base_agent.py`**

```python
from __future__ import annotations
import json
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.tool_registry import ToolRegistry
from src.agents.shared_memory import SharedMemory

class BaseAgent:
    def __init__(self, provider: BaseLLMProvider, registry: ToolRegistry, name: str):
        self._provider = provider
        self._registry = registry
        self._name = name

    def _step_cap_reached(self, memory: SharedMemory) -> bool:
        from src.config.settings import settings
        return memory.step_count >= settings.agent_max_steps

    async def _dispatch_tool_calls(
        self, tool_calls: list[dict], memory: SharedMemory
    ) -> list[dict]:
        results = []
        for tc in tool_calls:
            fn = tc["function"]
            inputs = json.loads(fn["arguments"])
            result = await self._registry.dispatch(fn["name"], inputs, memory)
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        return results
```

- [ ] **Step 4: Write `src/agents/executor_agent.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from src.agents.base_agent import BaseAgent
from src.agents.shared_memory import SharedMemory, ClinicalFlag
from src.config.settings import settings

EXECUTOR_SYSTEM_PROMPT = """You are a clinical data extraction agent producing a discharge summary draft.

HARD RULES — violation is not permitted:
1. NEVER infer, assume, or fill in missing clinical facts
2. If a field is absent → call flag_for_clinician_review with severity MISSING
3. If a result is pending → call flag_for_clinician_review with severity PENDING
4. If two documents disagree → call detect_conflicts — do NOT pick one
5. Compare admission and discharge medications → call reconcile_medications

WORKFLOW (follow in order):
1. Call read_pdf for EVERY PDF file in the patient folder
2. Call extract_section for each required field:
   patient_demographics, admission_date, discharge_date, principal_diagnosis,
   secondary_diagnoses, hospital_course, procedures, discharge_medications,
   allergies, follow_up_instructions, pending_results, discharge_condition
3. Call reconcile_medications
4. Call drug_interaction_lookup with the discharge medication names
5. When complete, respond with the text COMPILE and nothing else

You are done when you have processed all PDFs and all 12 sections.
"""

class ExecutorAgent(BaseAgent):

    async def run(self, patient_dir: str, memory: SharedMemory) -> None:
        pdf_paths = list(Path(patient_dir).glob("*.pdf"))
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Patient folder: {patient_dir}\n"
                f"PDF files found: {[str(p) for p in pdf_paths]}\n"
                "Begin processing."
            )},
        ]
        await self._run_loop(messages, memory)

    async def address_feedback(self, issues: list[str], memory: SharedMemory) -> None:
        draft_snapshot = json.dumps(memory.draft or {}, indent=2)[:3000]
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The critic identified these issues in your draft:\n"
                + "\n".join(f"- {issue}" for issue in issues)
                + f"\n\nCurrent draft:\n{draft_snapshot}\n\n"
                "Address each issue. Use flag_for_clinician_review if you cannot resolve it from the source documents."
            )},
        ]
        await self._run_loop(messages, memory)

    async def _run_loop(self, messages: list[dict], memory: SharedMemory) -> None:
        while not self._step_cap_reached(memory):
            content, tool_calls, _ = await self._provider.stream_complete(
                messages=messages,
                tools=self._registry.get_openai_definitions(),
                memory=memory,
                agent=self._name,
                action="plan_and_act",
                inputs={"step": memory.step_count},
            )

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })
                tool_results = await self._dispatch_tool_calls(tool_calls, memory)
                messages.extend(tool_results)
            else:
                self._compile_draft(memory)
                return

        memory.flags.append(ClinicalFlag(
            field="agent_control",
            reason=f"Step cap ({settings.agent_max_steps}) reached — draft may be incomplete",
            severity="MISSING",
            source_docs=[],
        ))
        self._compile_draft(memory)

    def _compile_draft(self, memory: SharedMemory) -> None:
        memory.draft = {
            section: field.model_dump()
            for section, field in memory.extracted_sections.items()
        }
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/agents/test_executor_agent.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/agents/base_agent.py src/agents/executor_agent.py tests/agents/test_executor_agent.py
git commit -m "feat: BaseAgent and ExecutorAgent with step cap and tool loop"
```

---

## Task 13: CriticAgent

**Files:**
- Create: `src/agents/critic_agent.py`
- Create: `tests/agents/test_critic_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_critic_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.critic_agent import CriticAgent, CriticFeedback
from src.agents.shared_memory import SharedMemory, ExtractedField, TraceStep
from datetime import datetime

def make_step():
    return TraceStep(
        step_id=0, agent="critic", reasoning="", action="review",
        inputs={}, result="", tokens_in=10, tokens_out=5,
        latency_ms=50.0, timestamp=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_critic_returns_approved():
    provider = MagicMock()
    provider.stream_complete = AsyncMock(return_value=("APPROVED", None, make_step()))

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {"principal_diagnosis": {"value": "Gastroenteritis", "confidence": "found"}}

    feedback = await agent.review(mem)
    assert feedback.approved is True
    assert feedback.issues == []

@pytest.mark.asyncio
async def test_critic_returns_issues():
    provider = MagicMock()
    provider.stream_complete = AsyncMock(return_value=(
        "Issues found:\n- principal_diagnosis has no raw_quote\n- allergies section missing",
        None, make_step()
    ))

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {}

    feedback = await agent.review(mem)
    assert feedback.approved is False
    assert len(feedback.issues) >= 2

@pytest.mark.asyncio
async def test_critic_calls_tools_before_verdict():
    tool_call = [{
        "id": "c1", "type": "function",
        "function": {"name": "flag_for_clinician_review",
                     "arguments": '{"field": "allergies", "reason": "missing", "severity": "MISSING"}'}
    }]
    call_count = 0
    async def stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("", tool_call, make_step())
        return ("APPROVED", None, make_step())

    provider = MagicMock()
    provider.stream_complete = stream_side_effect

    registry = MagicMock()
    registry.get_openai_definitions.return_value = []
    registry.dispatch = AsyncMock(return_value="[FLAGGED]")

    agent = CriticAgent(provider, registry, "critic")
    mem = SharedMemory(patient_id="p001")
    mem.draft = {}

    feedback = await agent.review(mem)
    assert feedback.approved is True
    registry.dispatch.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agents/test_critic_agent.py -v
```

- [ ] **Step 3: Write `src/agents/critic_agent.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass, field
from src.agents.base_agent import BaseAgent
from src.agents.shared_memory import SharedMemory

CRITIC_SYSTEM_PROMPT = """You are a clinical safety reviewer. The Executor has produced a discharge summary draft.

YOUR CHECKS:
1. Fabrication: any field with confidence="found" but raw_quote is null → flag it
2. Required sections completely absent from the draft → flag each
3. Medication reconciliation flags not raised for undocumented changes
4. Conflicts between documents not flagged
5. Drug interaction safety warnings not surfaced

If issues found: use detect_conflicts or flag_for_clinician_review for each issue, then
list ALL issues as bullet points starting with "Issues found:"

If everything is acceptable: respond with exactly "APPROVED" and nothing else.

You are the last line of defence before this draft reaches a clinician. Be strict.
"""

REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]

@dataclass
class CriticFeedback:
    approved: bool
    issues: list[str] = field(default_factory=list)

class CriticAgent(BaseAgent):

    async def review(self, memory: SharedMemory) -> CriticFeedback:
        draft_text = json.dumps(memory.draft or {}, indent=2)[:4000]
        flags_text = json.dumps([f.model_dump() for f in memory.flags], indent=2)[:2000]

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Draft:\n{draft_text}\n\n"
                f"Existing flags:\n{flags_text}\n\n"
                f"Required sections: {REQUIRED_SECTIONS}"
            )},
        ]

        while not self._step_cap_reached(memory):
            content, tool_calls, _ = await self._provider.stream_complete(
                messages=messages,
                tools=self._registry.get_openai_definitions(),
                memory=memory,
                agent=self._name,
                action="review",
                inputs={"handoff_round": memory.handoff_round},
            )

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })
                tool_results = await self._dispatch_tool_calls(tool_calls, memory)
                messages.extend(tool_results)
            else:
                approved = content.strip().upper() == "APPROVED"
                if approved:
                    return CriticFeedback(approved=True)
                issues = [
                    line.lstrip("-• ").strip()
                    for line in content.splitlines()
                    if line.strip() and not line.strip().startswith("Issues found")
                ]
                return CriticFeedback(approved=False, issues=issues)

        return CriticFeedback(
            approved=False,
            issues=["Step cap reached during critic review — manual review required"]
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/agents/test_critic_agent.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/agents/critic_agent.py tests/agents/test_critic_agent.py
git commit -m "feat: CriticAgent with tool loop and CriticFeedback"
```

---

## Task 14: Orchestrator

**Files:**
- Create: `src/agents/orchestrator.py`
- Create: `tests/agents/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_orchestrator.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import DischargeAgentOrchestrator
from src.agents.critic_agent import CriticFeedback
from src.agents.shared_memory import SharedMemory

@pytest.mark.asyncio
async def test_orchestrator_approves_on_first_review():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(return_value=CriticFeedback(approved=True))

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 3
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir="data/patient2", patient_id="p002")

    assert memory.critic_approved is True
    assert memory.handoff_round == 1
    executor.address_feedback.assert_not_called()

@pytest.mark.asyncio
async def test_orchestrator_retries_after_feedback():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(side_effect=[
        CriticFeedback(approved=False, issues=["allergies missing"]),
        CriticFeedback(approved=True),
    ])

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 3
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir="data/patient2", patient_id="p002")

    assert memory.critic_approved is True
    assert memory.handoff_round == 2
    executor.address_feedback.assert_called_once_with(["allergies missing"], memory)

@pytest.mark.asyncio
async def test_orchestrator_flags_when_never_approved():
    executor = MagicMock()
    executor.run = AsyncMock()
    executor.address_feedback = AsyncMock()

    critic = MagicMock()
    critic.review = AsyncMock(
        return_value=CriticFeedback(approved=False, issues=["unresolved issue"])
    )

    with patch("src.agents.orchestrator.settings") as ms:
        ms.agent_max_handoff_rounds = 2
        orch = DischargeAgentOrchestrator(executor, critic)
        memory = await orch.run(patient_dir=".", patient_id="p001")

    assert memory.critic_approved is False
    never_approved_flags = [f for f in memory.flags if "did not approve" in f.reason.lower()]
    assert len(never_approved_flags) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agents/test_orchestrator.py -v
```

- [ ] **Step 3: Write `src/agents/orchestrator.py`**

```python
from __future__ import annotations
from src.agents.executor_agent import ExecutorAgent
from src.agents.critic_agent import CriticAgent
from src.agents.shared_memory import SharedMemory, ClinicalFlag
from src.config.settings import settings

class DischargeAgentOrchestrator:

    def __init__(self, executor: ExecutorAgent, critic: CriticAgent):
        self._executor = executor
        self._critic = critic

    async def run(self, patient_dir: str, patient_id: str) -> SharedMemory:
        memory = SharedMemory(patient_id=patient_id)

        await self._executor.run(patient_dir=patient_dir, memory=memory)

        for _ in range(settings.agent_max_handoff_rounds):
            memory.handoff_round += 1
            feedback = await self._critic.review(memory)

            if feedback.approved:
                memory.critic_approved = True
                break

            await self._executor.address_feedback(feedback.issues, memory)

        if not memory.critic_approved:
            memory.flags.append(ClinicalFlag(
                field="review_status",
                reason=(
                    f"Critic did not approve after {memory.handoff_round} handoff round(s) — "
                    "full clinician review required before use"
                ),
                severity="MISSING",
                source_docs=[],
            ))

        return memory
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/agents/test_orchestrator.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "feat: DischargeAgentOrchestrator with handoff loop and step guards"
```

---

## Task 15: Chainlit UI

**Files:**
- Create: `chainlit_app.py`

- [ ] **Step 1: Write `chainlit_app.py`**

```python
import json
import chainlit as cl
from pathlib import Path
from src.config.settings import settings
from src.providers.azure.llm_provider import AsyncAzureLLMProvider
from src.agents.tools import build_executor_registry, build_critic_registry
from src.agents.executor_agent import ExecutorAgent
from src.agents.critic_agent import CriticAgent
from src.agents.orchestrator import DischargeAgentOrchestrator
from src.agents.shared_memory import TraceStep

REQUIRED_SECTIONS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "allergies",
    "follow_up_instructions", "pending_results", "discharge_condition",
]

SEVERITY_ICON = {
    "MISSING": "⚠️",
    "PENDING": "⏳",
    "CONFLICT": "🔴",
    "RECONCILIATION_NEEDED": "💊",
    "SAFETY": "🚨",
}

CONFIDENCE_BADGE = {"found": "✓", "pending": "⏳", "missing": "⚠"}

@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "**Discharge Summary Agent**\n\n"
            "Provide the path to a patient folder containing source-note PDFs.\n\n"
            "Example: `data/patient2`"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    patient_dir = message.content.strip()

    if not Path(patient_dir).exists():
        await cl.Message(content=f"Path not found: `{patient_dir}`").send()
        return

    # Active Chainlit step for streaming tokens
    active_step: dict = {"ref": None}

    async def token_callback(chunk: str):
        if active_step["ref"]:
            await active_step["ref"].stream_token(chunk)

    async def emit_callback(step: TraceStep):
        agent_label = "Executor" if step.agent == "executor" else "Critic"
        async with cl.Step(name=f"[{agent_label}] {step.action}") as cl_step:
            cl_step.input = (
                f"**Reasoning:** {step.reasoning[:300]}\n\n"
                f"**Inputs:** `{json.dumps(step.inputs)}`"
            )
            cl_step.output = (
                f"**Result:** {step.result[:400]}\n\n"
                f"Tokens: {step.tokens_in} in / {step.tokens_out} out — "
                f"Latency: {step.latency_ms:.0f}ms"
            )
            await cl_step.update()

    provider = AsyncAzureLLMProvider(emit_callback=emit_callback)
    executor = ExecutorAgent(provider, build_executor_registry(provider), "executor")
    critic = CriticAgent(provider, build_critic_registry(provider), "critic")
    orchestrator = DischargeAgentOrchestrator(executor, critic)

    patient_id = Path(patient_dir).name
    await cl.Message(content=f"Processing patient `{patient_id}`...").send()

    memory = await orchestrator.run(patient_dir=patient_dir, patient_id=patient_id)

    await _render_summary(memory)
    await _render_flags(memory)
    await _render_stats(memory)

async def _render_summary(memory):
    sections = memory.draft or {}
    lines = [
        f"## Discharge Summary Draft — `{memory.patient_id}`",
        f"*Critic approved: {'Yes ✓' if memory.critic_approved else 'No — full clinician review required ⚠️'}*",
        "---",
    ]
    for key in REQUIRED_SECTIONS:
        field = sections.get(key, {})
        value = field.get("value") or "**[MISSING — flagged for clinician review]**"
        confidence = field.get("confidence", "missing")
        badge = CONFIDENCE_BADGE.get(confidence, "?")
        label = key.replace("_", " ").title()
        lines.append(f"**{label}** {badge}\n{value}\n")

    await cl.Message(content="\n".join(lines)).send()

async def _render_flags(memory):
    if not memory.flags:
        await cl.Message(content="No clinical flags raised.").send()
        return
    lines = [f"## Clinical Flags ({len(memory.flags)}) — Requires Clinician Review", ""]
    for f in memory.flags:
        icon = SEVERITY_ICON.get(f.severity, "⚠️")
        lines.append(f"{icon} **{f.severity}** | `{f.field}` — {f.reason}")
    await cl.Message(content="\n".join(lines)).send()

async def _render_stats(memory):
    total_tokens = sum(s.tokens_in + s.tokens_out for s in memory.trace)
    avg_latency = (
        sum(s.latency_ms for s in memory.trace) / len(memory.trace)
        if memory.trace else 0
    )
    await cl.Message(content=(
        f"**Run stats** — "
        f"Steps: {memory.step_count} | "
        f"Handoff rounds: {memory.handoff_round} | "
        f"Total tokens: {total_tokens:,} | "
        f"Avg latency: {avg_latency:.0f}ms | "
        f"Conflicts: {len(memory.conflicts)} | "
        f"Flags: {len(memory.flags)}"
    )).send()
```

- [ ] **Step 2: Test the UI manually**

```bash
conda activate healthcare
chainlit run chainlit_app.py
```

Open `http://localhost:8000`. Type `data/patient 2 (1)_260603_095051` and verify:
- Steps appear in real-time with reasoning + token counts
- Final summary has `⚠ CONFLICT` for the DKA vs Gastroenteritis diagnosis conflict
- `⏳ PENDING` for the urine culture result
- `💊 RECONCILIATION_NEEDED` for medications with missing dosages

- [ ] **Step 3: Commit**

```bash
git add chainlit_app.py
git commit -m "feat: Chainlit UI with real-time streaming steps, flags panel, and run stats"
```

---

## Task 16: Synthetic Patient 1 Data

**Files:**
- Create: `scripts/generate_patient1.py`
- Create: `data/patient1/` (4 PDF files)

- [ ] **Step 1: Write `scripts/generate_patient1.py`**

```python
"""
Generate synthetic patient 1 PDF data with intentional conflicts and missing fields.
Run: python scripts/generate_patient1.py
"""
import os
import fitz

os.makedirs("data/patient1", exist_ok=True)

def make_pdf(path: str, content: str):
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for line in content.strip().splitlines():
        page.insert_text((50, y), line, fontsize=11)
        y += 18
        if y > 750:
            page = doc.new_page()
            y = 80
    doc.save(path)
    doc.close()
    print(f"Created: {path}")

# Admission note
make_pdf("data/patient1/admission_note.pdf", """
ADMISSION NOTE

Patient: John Smith | MRN: 00124 | DOB: 1965-03-15
Admission Date: 2026-05-20
Ward: General Medicine

PRESENTING COMPLAINT:
Chest pain and shortness of breath for 2 days.

HISTORY:
Known hypertensive. On Amlodipine 5mg OD and Aspirin 75mg OD.
No known drug allergies.

DIAGNOSIS ON ADMISSION:
Suspected NSTEMI — troponin pending.

PHYSICAL EXAMINATION:
PR: 92/min, BP: 158/94 mmHg, RR: 18/min, SpO2: 96% on room air
Chest: Bilateral air entry, mild crepitations at left base.
CVS: S1 S2 heard, no murmurs.

PLAN:
Admit, ECG, troponin serial, echo, cardiology review.
""")

# Progress note — intentional conflict: calls it ACS, not NSTEMI
make_pdf("data/patient1/progress_note_day2.pdf", """
PROGRESS NOTE — Day 2

Date: 2026-05-21

Patient stable overnight. Troponin: 0.8 ng/mL (elevated).
Echo: Mild LV dysfunction, EF 45%.

WORKING DIAGNOSIS: Acute Coronary Syndrome (ACS)
(Note: Admission note called this NSTEMI — please reconcile)

MEDICATIONS TODAY:
- Aspirin 75mg OD (continued)
- Atorvastatin 40mg ON (NEW — no reason documented in chart)
- Amlodipine 5mg OD (continued)
- Heparin infusion (started for anticoagulation)

Cardiology reviewed. Plan: proceed to angiography tomorrow.
""")

# Lab results — pending culture
make_pdf("data/patient1/lab_results.pdf", """
LABORATORY RESULTS

Patient: John Smith | MRN: 00124

Troponin I:      0.8 ng/mL (HIGH)  [Ref: <0.04]
CBC:             WBC 11.2, Hb 13.4, Plt 210 — normal
Serum Creatinine: 1.1 mg/dL (normal)
HbA1c:           7.2% (mildly elevated)
Lipid Panel:
  LDL: 138 mg/dL (elevated)
  HDL: 42 mg/dL

Blood Culture:   Sent 2026-05-20 — REPORT AWAITED
""")

# Discharge medication record
make_pdf("data/patient1/discharge_medications.pdf", """
DISCHARGE MEDICATION RECORD

Patient: John Smith | MRN: 00124
Discharge Date: 2026-05-23

CONDITION AT DISCHARGE: Stable, ambulatory.

DISCHARGE MEDICATIONS:
1. Aspirin 75mg OD — continue (was on admission)
2. Atorvastatin 40mg ON — continue (started during admission)
3. Amlodipine 5mg OD — continue (was on admission)
4. Bisoprolol 2.5mg OD — NEW (no reason documented)
5. Ramipril 5mg OD — NEW (no reason documented)

STOPPED: Heparin infusion (IV — discontinued prior to discharge)

FOLLOW-UP:
Cardiology OPD: 2026-06-03
Repeat Echo: 2026-06-10
Blood culture report: collect when available.

ALLERGIES: No known drug allergies.
""")

print("\nPatient 1 data generated in data/patient1/")
print("Known test scenarios:")
print("  - CONFLICT: admission=NSTEMI vs progress=ACS")
print("  - PENDING: blood culture report awaited")
print("  - RECONCILIATION_NEEDED: Bisoprolol + Ramipril added, no reason; Atorvastatin no reason")
```

- [ ] **Step 2: Run the generator**

```bash
conda activate healthcare
python scripts/generate_patient1.py
```

Expected: 4 PDF files created in `data/patient1/`

- [ ] **Step 3: Run the agent against patient 1 via Chainlit**

```bash
chainlit run chainlit_app.py
```

Type `data/patient1` and verify:
- `🔴 CONFLICT` for NSTEMI vs ACS diagnosis
- `⏳ PENDING` for blood culture
- `💊 RECONCILIATION_NEEDED` for Bisoprolol, Ramipril (no reason), Atorvastatin (no reason)
- Heparin stops with no reason → `RECONCILIATION_NEEDED`

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_patient1.py data/patient1/
git commit -m "feat: synthetic patient 1 data with known conflicts and pending results"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Run both patients end-to-end**

```bash
chainlit run chainlit_app.py
```

Test `data/patient1` and `data/patient 2 (1)_260603_095051` (or the actual patient 2 folder). Confirm all known conflicts and pending results surface correctly.

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: Part 1 complete — discharge summary agent with Executor+Critic, streaming, Chainlit UI"
```

---

## Checklist Against Spec Requirements

| Requirement | Task |
|---|---|
| Real agent loop (plan + replan) | Tasks 12–14 |
| PDF ingestion (PyMuPDF + OCR) | Task 7 |
| No fabrication guardrail | Tasks 8, 13 (raw_quote anchor + Critic) |
| Handle pending/missing data | Tasks 8, 10 |
| Medication reconciliation | Task 9 |
| Handle conflicting information | Task 9 |
| Tool use + agent decides when | Tasks 4, 12, 13 |
| Robust failure handling | Tasks 6, 7, 12 |
| Hard step/iteration cap | Tasks 12, 13, 14 |
| Observability (trace per step) | Tasks 5, 6 |
| Chainlit streaming demo | Tasks 6, 15 |
| Two patients for video demo | Task 16 |
