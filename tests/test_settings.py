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
        from src.config.settings import Settings
        s = Settings(_env_file=None)  # bypass .env file entirely — hermetic test
        assert s.azure_openai_endpoint == "https://example.azure.com/"
        assert s.azure_llm_deployment == "gpt-4o"
        assert s.azure_llm_api_version == "2025-01-01-preview"
        assert s.azure_openai_api_key.get_secret_value() == "test-key-123"  # SecretStr: use get_secret_value()
        assert s.agent_max_steps == 50
        assert s.agent_max_handoff_rounds == 3
        assert s.pdf_ocr_fallback_min_chars == 50
