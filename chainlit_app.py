import json
from pathlib import Path

import chainlit as cl

from src.agents.critic_agent import CriticAgent
from src.agents.executor_agent import ExecutorAgent
from src.agents.orchestrator import DischargeAgentOrchestrator
from src.agents.shared_memory import SharedMemory, SourceDocument, TraceStep
from src.agents.tools import build_critic_registry, build_executor_registry
from src.agents.tools.read_pdf import ReadPDFTool
from src.providers.azure.llm_provider import AsyncAzureLLMProvider
from src.utils.pdf_cache import PDFCache
from src.utils.output_writer import save_output, REQUIRED_SECTIONS, SEVERITY_ICON, CONFIDENCE_BADGE


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "## Discharge Summary Agent\n\n"
            "Upload a patient PDF using the **📎 attachment button** in the input box below, then hit send."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Accept uploaded PDF files
    pdf_elements = [
        el for el in (message.elements or [])
        if getattr(el, "mime", "") == "application/pdf"
        or (getattr(el, "name", "") or "").lower().endswith(".pdf")
    ]

    if not pdf_elements:
        await cl.Message(
            content="Please upload a PDF file using the 📎 attachment button, then send."
        ).send()
        return

    uploaded = pdf_elements[0]
    pdf_path = uploaded.path          # server-side temp path
    patient_id = Path(uploaded.name).stem
    patient_dir = str(Path(pdf_path).parent)
    pdf_files = [pdf_path]

    if not pdf_files:
        await cl.Message(
            content=f"❌ No PDF files found in `{input_path}`"
        ).send()
        return

    await cl.Message(content=f"🏥 Processing patient **{patient_id}**…").send()

    cache = PDFCache()

    async def emit_callback(step: TraceStep):
        agent_label = "Executor" if step.agent == "executor" else "Critic"
        icon = "🔧" if step.agent == "executor" else "🔍"
        result_text = step.result[:500] + ("…" if len(step.result) > 500 else "")

        async with cl.Step(
            name=f"{icon} [{agent_label}] step {step.step_id} — {step.action}"
        ) as cl_step:
            cl_step.input = (
                f"**Reasoning:** {step.reasoning}\n\n"
                f"**Inputs:** `{json.dumps(step.inputs, default=str)}`"
            )
            cl_step.output = (
                f"{result_text}\n\n"
                f"---\n"
                f"🔢 Tokens: `{step.tokens_in}` in / `{step.tokens_out}` out  "
                f"⏱ Latency: `{step.latency_ms:.0f}ms`"
            )

    provider = AsyncAzureLLMProvider(emit_callback=emit_callback)

    # Phase 1: pre-extract all PDFs with visible progress (warms cache)
    pre_documents = await _pre_extract_pdfs(pdf_files, provider, cache)

    # Phase 2: run the agent loop (read_pdf calls will be instant cache hits)
    executor = ExecutorAgent(provider, build_executor_registry(provider, cache=cache), "executor")
    critic = CriticAgent(provider, build_critic_registry(provider), "critic")
    orchestrator = DischargeAgentOrchestrator(executor, critic)

    try:
        memory = await orchestrator.run(
            patient_dir=patient_dir,
            patient_id=patient_id,
            pdf_files=pdf_files,
            pre_documents=pre_documents,
        )
    except Exception as e:
        await cl.Message(
            content=f"❌ Agent error: `{type(e).__name__}: {e}`"
        ).send()
        raise

    await _render_summary(memory)
    await _render_flags(memory)
    await _render_stats(memory)

    output_path = save_output(memory)
    await cl.Message(
        content=(
            f"💾 Saved to `{output_path}`\n\n"
            f"- `discharge_summary.md` — human-readable summary\n"
            f"- `discharge_summary.json` — structured data\n"
            f"- `trace.json` — step-by-step agent trace\n"
            f"- `flags.json` — clinical flags & conflicts"
        )
    ).send()


async def _pre_extract_pdfs(
    pdf_paths: list[str],
    provider: AsyncAzureLLMProvider,
    cache: PDFCache,
) -> list[SourceDocument]:
    tool = ReadPDFTool(provider, cache=cache)
    pre_memory = SharedMemory(patient_id="__pre_extract__")

    async with cl.Step(name="📄 Extracting PDF content") as step:
        step.output = ""
        for pdf_path in pdf_paths:
            result = await tool.execute({"path": pdf_path}, pre_memory)
            step.output += f"\n✓ {Path(pdf_path).name} — {result}"

    return pre_memory.source_documents


async def _render_summary(memory: SharedMemory):
    sections = memory.draft or {}
    lines = [
        f"## Discharge Summary — `{memory.patient_id}`",
        f"*Critic approved: {'**Yes ✓**' if memory.critic_approved else '**No** — full clinician review required ⚠️'}*",
        "",
        "---",
        "",
    ]
    for key in REQUIRED_SECTIONS:
        field = sections.get(key)
        if isinstance(field, dict):
            value = field.get("value") or "**[MISSING — flagged for clinician review]**"
            confidence = field.get("confidence", "missing")
        else:
            value = "**[MISSING — flagged for clinician review]**"
            confidence = "missing"
        badge = CONFIDENCE_BADGE.get(confidence, "?")
        label = key.replace("_", " ").title()
        lines.append(f"**{label}** `{badge}`")
        lines.append(value)
        lines.append("")

    await cl.Message(content="\n".join(lines)).send()


async def _render_flags(memory: SharedMemory):
    if not memory.flags:
        await cl.Message(content="✅ No clinical flags raised.").send()
        return
    lines = [
        f"## Clinical Flags ({len(memory.flags)}) — Requires Clinician Review",
        "",
    ]
    for f in memory.flags:
        icon = SEVERITY_ICON.get(f.severity, "⚠️")
        lines.append(f"{icon} **{f.severity}** | `{f.field}` — {f.reason}")
    await cl.Message(content="\n".join(lines)).send()


async def _render_stats(memory: SharedMemory):
    total_tokens = sum(s.tokens_in + s.tokens_out for s in memory.trace)
    avg_latency = (
        sum(s.latency_ms for s in memory.trace) / len(memory.trace)
        if memory.trace else 0
    )
    await cl.Message(
        content=(
            f"**Run Stats**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Steps | {memory.step_count} |\n"
            f"| Handoff rounds | {memory.handoff_round} |\n"
            f"| Total tokens | {total_tokens:,} |\n"
            f"| Avg latency | {avg_latency:.0f}ms |\n"
            f"| Conflicts | {len(memory.conflicts)} |\n"
            f"| Flags | {len(memory.flags)} |\n"
            f"| Critic approved | {'Yes ✓' if memory.critic_approved else 'No ⚠️'} |"
        )
    ).send()
