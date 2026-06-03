# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def set_dummy_azure_env():
    """Prevent ValidationError when src.config.settings is imported at collection time."""
    defaults = {
        "AZURE_OPENAI_ENDPOINT": "https://dummy.azure.com/",
        "AZURE_LLM_DEPLOYMENT": "gpt-4o",
        "AZURE_LLM_API_VERSION": "2025-01-01-preview",
        "AZURE_OPENAI_API_KEY": "dummy-key",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
