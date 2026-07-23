"""
JARVIS Application Settings.

1. Why this module exists:
   Centralizes configuration management using Pydantic BaseSettings and python-dotenv.
   Ensures type safety, validation, default values, and prevents hardcoded secrets.

2. How it fits into the architecture:
   Part of the Infrastructure / Configuration layer. Supplies configuration parameters
   to services, core orchestrator, tools, and voice lifecycle.

3. Which future modules will interact with it:
   - backend.services.* (Service configurations, API endpoints)
   - backend.ai.* (LLM provider API keys, model selections)
   - backend.voice.* (STT/TTS configuration settings)
   - backend.core.orchestrator (System operational modes)

4. Common mistakes to avoid:
   - Hardcoding default secrets or API keys in class attributes.
   - Accessing os.environ directly instead of using get_settings().

5. Possible future improvements:
   - Encrypted secrets management via key vault / AWS Secrets Manager integration.
   - Per-user setting overrides stored in database.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """JARVIS System Configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core Application Settings
    env: Literal["development", "staging", "production"] = Field(
        default="development", description="Execution environment mode"
    )
    debug: bool = Field(default=True, description="Enable debug logging and diagnostics")
    log_level: str = Field(default="INFO", description="Logging level threshold")

    # Assistant Profile
    assistant_name: str = Field(default="JARVIS", description="Display name of assistant")
    wake_word: str = Field(default="jarvis", description="Wake word phrase for voice activation")

    # Third-Party Service Configuration (V1.0 Placeholders)
    weather_api_key: str | None = Field(default=None, description="OpenWeatherMap API Key")
    news_api_key: str | None = Field(default=None, description="NewsAPI Key")

    # AI Provider Configuration
    llm_provider: str = Field(default="mock", description="Selected AI LLM Provider")
    llm_model: str = Field(default="mock-v1", description="Model name identifier")
    llm_api_key: str | None = Field(default=None, description="LLM Provider API Key")

    # Voice Engine Configuration
    stt_engine: str = Field(default="mock", description="Speech-To-Text engine type")
    tts_engine: str = Field(default="mock", description="Text-To-Speech engine type")


@lru_cache
def get_settings() -> Settings:
    """Singleton getter for application settings with caching."""
    return Settings()
