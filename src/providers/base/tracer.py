from __future__ import annotations
import time
from datetime import datetime, timezone
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
        if self._start is None:
            raise RuntimeError("LLMCallTracer.record() called outside a 'with' block")
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
            timestamp=datetime.now(timezone.utc),
        )
        self._memory.trace.append(step)
        self._memory.step_count += 1
        if self._emit_callback:
            await self._emit_callback(step)
        return step

    async def record_failure(self, error: str) -> TraceStep:
        if self._start is None:
            raise RuntimeError("LLMCallTracer.record() called outside a 'with' block")
        latency_ms = round((time.perf_counter() - self._start) * 1000, 2)
        step = TraceStep(
            step_id=self._memory.step_count,
            agent=self._agent,
            reasoning="LLM call failed",
            action=self._action,
            inputs=self._inputs,
            result=f"[LLM_CALL_FAILED] {error}",
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
        )
        self._memory.trace.append(step)
        self._memory.step_count += 1
        if self._emit_callback:
            await self._emit_callback(step)
        return step
