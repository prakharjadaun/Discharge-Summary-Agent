import os
import pytest
from unittest.mock import patch

def test_settings_loads_from_env():
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://example.azure.com/",
        "AZURE_LLM_DEPLOYMENT": "gpt-4o",
        "AZURE_LLM_API_VERSION": "2025-01-01-preview",
        "AZURE_OPENAI_API_KEY": "test-key-123",
    }
    with patch.dict(os.environ, env, clear=True):
        from importlib import reload
        import src.config.settings as s
        reload(s)
        assert s.settings.azure_llm_deployment == "gpt-4o"
        assert s.settings.agent_max_steps == 20
        assert s.settings.agent_max_handoff_rounds == 3
        assert s.settings.pdf_ocr_fallback_min_chars == 50
