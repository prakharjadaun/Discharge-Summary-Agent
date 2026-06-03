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
