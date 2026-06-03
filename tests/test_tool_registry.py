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

async def test_registry_dispatch_success():
    registry = ToolRegistry()
    registry.register(EchoTool())
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("echo", {"message": "hello"}, mem)
    assert result == "ECHO: hello"

async def test_registry_dispatch_not_found():
    registry = ToolRegistry()
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("nonexistent", {}, mem)
    assert "[TOOL_NOT_FOUND: nonexistent]" in result

async def test_registry_dispatch_handles_exception():
    registry = ToolRegistry()
    registry.register(CrashTool())
    mem = SharedMemory(patient_id="test")
    result = await registry.dispatch("crash", {}, mem)
    assert result == "[TOOL_ERROR: crash \u2014 intentional crash]"


def test_registry_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())

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
