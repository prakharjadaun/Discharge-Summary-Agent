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


async def test_stream_complete_text_response():
    from src.providers.azure.llm_provider import AsyncAzureLLMProvider

    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI") as MockClient:
        instance = MockClient.return_value
        chunks = make_stream_chunks(content="test response")
        instance.chat.completions.create = AsyncMock(return_value=async_iter(chunks))

        mem = SharedMemory(patient_id="test")
        provider = AsyncAzureLLMProvider.__new__(AsyncAzureLLMProvider)
        provider._client = instance
        provider._deployment = "gpt-4o"
        provider._emit_callback = None

        tokens = []

        async def collect(chunk):
            tokens.append(chunk)

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


async def test_stream_complete_tool_call():
    from src.providers.azure.llm_provider import AsyncAzureLLMProvider

    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI") as MockClient:
        instance = MockClient.return_value
        chunks = make_stream_chunks(
            content=None,
            tool_name="read_pdf",
            tool_args='{"path": "test.pdf"}'
        )
        instance.chat.completions.create = AsyncMock(return_value=async_iter(chunks))

        mem = SharedMemory(patient_id="test")
        provider = AsyncAzureLLMProvider.__new__(AsyncAzureLLMProvider)
        provider._client = instance
        provider._deployment = "gpt-4o"
        provider._emit_callback = None

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


async def test_stream_complete_retries_and_succeeds():
    """Provider retries on APIError and succeeds on the third attempt."""
    from src.providers.azure.llm_provider import AsyncAzureLLMProvider
    from openai import APIError

    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI"):
        mem = SharedMemory(patient_id="test")
        provider = AsyncAzureLLMProvider.__new__(AsyncAzureLLMProvider)
        provider._deployment = "gpt-4o"
        provider._emit_callback = None

        call_count = 0
        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("rate limit", request=MagicMock(), body={})
            return async_iter(make_stream_chunks(content="recovered"))

        provider._client = MagicMock()
        provider._client.chat.completions.create = fake_create

        with patch("src.providers.azure.llm_provider.asyncio.sleep", new_callable=AsyncMock):
            content, tool_calls, step = await provider.stream_complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                memory=mem,
                agent="executor",
                action="plan",
                inputs={},
            )

        assert content == "recovered"
        assert call_count == 3


async def test_stream_complete_all_retries_fail():
    """Provider returns [LLM_CALL_FAILED] after all 3 attempts fail."""
    from src.providers.azure.llm_provider import AsyncAzureLLMProvider
    from openai import APITimeoutError

    with patch("src.providers.azure.llm_provider.AsyncAzureOpenAI"):
        mem = SharedMemory(patient_id="test")
        provider = AsyncAzureLLMProvider.__new__(AsyncAzureLLMProvider)
        provider._deployment = "gpt-4o"
        provider._emit_callback = None

        async def always_fail(**kwargs):
            raise APITimeoutError(request=MagicMock())

        provider._client = MagicMock()
        provider._client.chat.completions.create = always_fail

        with patch("src.providers.azure.llm_provider.asyncio.sleep", new_callable=AsyncMock):
            content, tool_calls, step = await provider.stream_complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                memory=mem,
                agent="executor",
                action="plan",
                inputs={},
            )

        assert content == "[LLM_CALL_FAILED]"
        assert tool_calls is None
        assert "[LLM_CALL_FAILED]" in step.result
