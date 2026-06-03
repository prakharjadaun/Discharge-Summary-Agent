from src.agents.tools.base import BaseTool
from src.agents.shared_memory import SharedMemory


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
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
            return f"[TOOL_ERROR: {tool_name} \u2014 {str(e)}]"
