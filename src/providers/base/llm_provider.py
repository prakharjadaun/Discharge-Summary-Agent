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

    async def vision_transcribe(self, image_b64: str) -> str:
        """Transcribe a page image (handwritten or printed) using vision.
        Override in provider implementations that support vision.
        """
        return ""
