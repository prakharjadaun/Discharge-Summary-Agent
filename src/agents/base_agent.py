from __future__ import annotations
import json
from typing import TYPE_CHECKING
from src.providers.base.llm_provider import BaseLLMProvider
from src.agents.shared_memory import SharedMemory

if TYPE_CHECKING:
    from src.agents.tool_registry import ToolRegistry


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
            try:
                inputs = json.loads(fn["arguments"])
            except (json.JSONDecodeError, TypeError):
                inputs = {}
            result = await self._registry.dispatch(fn["name"], inputs, memory)
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        return results
