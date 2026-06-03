from __future__ import annotations
import asyncio
import types
from typing import Awaitable, Callable
from openai import AsyncAzureOpenAI, APITimeoutError, APIError
from src.providers.base.llm_provider import BaseLLMProvider
from src.providers.base.tracer import LLMCallTracer
from src.agents.shared_memory import SharedMemory, TraceStep
from src.config.settings import settings


_MAX_REASONING_PREVIEW = 200


class AsyncAzureLLMProvider(BaseLLMProvider):

    def __init__(
        self,
        emit_callback: Callable[[TraceStep], Awaitable[None]] | None = None,
    ):
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
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
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("stream_complete retry loop exhausted without returning")

    async def _do_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tracer: LLMCallTracer,
        token_callback: Callable[[str], Awaitable[None]] | None,
    ) -> tuple[str, list[dict] | None, TraceStep]:

        kwargs: dict = dict(
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
            content[:_MAX_REASONING_PREVIEW] if content
            else f"tool_call: {next(iter(tool_calls_acc.values()))['function']['name']}" if tool_calls_acc
            else "no content"
        )
        step = await tracer.record(reasoning=reasoning, response=fake_response)
        return content, tool_calls, step
