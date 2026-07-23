"""
JARVIS Application Settings & AI Configuration Models.

1. Why this module exists:
   Centralizes configuration management using Pydantic BaseSettings and python-dotenv.
   Ensures type safety, validation, default values, and encapsulates provider settings via `AIConfig`.

2. How it fits into the architecture:
   Part of Infrastructure / Configuration layer. Converts raw environment variables into structured `AIConfig` models.

3. Which future modules will interact with it:
   - `backend.ai.providers.gemini_provider.GeminiProvider`
   - `backend.core.startup.StartupManager`

4. Common mistakes to avoid:
   - Hardcoding default secrets or API keys in source code.

5. Possible future improvements:
   - Encrypted secrets vault integration.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIConfig(BaseModel):
    """Encapsulates AI LLM provider configuration settings."""

    provider: str = Field(default="mock", description="Selected AI Provider ('mock' or 'gemini')")
    model_name: str = Field(default="gemini-2.5-flash", description="Model name identifier")
    api_key: str | None = Field(default=None, description="API Key for target provider")
    temperature: float = Field(default=0.7, description="Generation temperature (0.0 to 1.0)")
    max_tokens: int = Field(default=1024, description="Maximum token generation limit")


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
    conversation_history_limit: int = Field(default=20, description="Max conversation turn history limit")

    # Third-Party Service Configuration
    weather_api_key: str | None = Field(default=None, description="OpenWeatherMap API Key")
    news_api_key: str | None = Field(default=None, description="NewsAPI Key")

    # AI Provider Configuration
    llm_provider: str = Field(default="gemini", description="Selected AI LLM Provider ('gemini' or 'mock')")
    llm_model: str = Field(default="gemini-2.5-flash", description="Model name identifier")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API Key")
    llm_api_key: str | None = Field(default=None, description="Generic LLM Provider API Key")
    llm_temperature: float = Field(default=0.7, description="Generation temperature placeholder")
    llm_max_tokens: int = Field(default=1024, description="Max tokens placeholder")

    # Voice Configuration Section Placeholder (Sprint 2.1 & Task 1)
    voice_provider: str = Field(default="pyttsx3", description="Selected TTS Provider")
    voice_rate: int = Field(default=200, description="Speech rate in words per minute")
    voice_volume: float = Field(default=1.0, description="Speech output volume (0.0 to 1.0)")
    voice_id: str | None = Field(default=None, description="Target TTS voice identifier")
    stt_provider: str = Field(default="faster_whisper", description="Selected STT Provider ('faster_whisper' or 'mock')")
    whisper_model: str = Field(default="base", description="Faster-Whisper model size ('tiny', 'base', 'small', 'medium', 'large')")
    stt_device: str = Field(default="cpu", description="Compute device for STT ('cpu' or 'cuda')")
    stt_compute_type: str = Field(default="int8", description="Quantization compute type ('int8', 'float16', 'float32')")
    wakeword_provider: str = Field(default="openwakeword", description="Selected wake word provider ('openwakeword' or 'mock')")
    wakeword_sensitivity: float = Field(default=0.5, description="Wake word detection threshold (0.0 to 1.0)")
    listen_timeout: float = Field(default=8.0, description="Active listening duration in seconds")
    post_response_timeout: float = Field(default=5.0, description="Continuous follow-up listening window duration in seconds")
    voice_cooldown_seconds: float = Field(default=1.0, description="Post-TTS cooldown pause in seconds")
    voice_mode: bool = Field(default=False, description="Default to hands-free voice interface mode on startup")

    def get_ai_config(self) -> AIConfig:
        """Constructs an AIConfig model from current settings."""
        api_key = self.gemini_api_key or self.llm_api_key
        return AIConfig(
            provider=self.llm_provider,
            model_name=self.llm_model,
            api_key=api_key,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )


@lru_cache
def get_settings() -> Settings:
    """Singleton getter for application settings with caching."""
    return Settings()
