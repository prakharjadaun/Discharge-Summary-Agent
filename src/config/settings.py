from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_llm_deployment: str
    azure_llm_api_version: str
    azure_openai_api_key: str

    agent_max_steps: int = 20
    agent_max_handoff_rounds: int = 3
    pdf_ocr_fallback_min_chars: int = 50

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()
