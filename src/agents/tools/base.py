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
