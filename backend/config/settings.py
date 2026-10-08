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
    wake_acknowledgement: str = Field(default="Yes, Anurag sir.", description="Spoken TTS acknowledgement phrase when wake word is detected")
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
    ocr_provider: str = Field(default="easyocr", description="Selected OCR Provider ('easyocr' or 'mock')")
    vision_provider: str = Field(default="gemini", description="Selected Vision Provider ('gemini' or 'mock')")

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
    wakeword_sensitivity: float = Field(default=0.35, description="Wake word detection threshold (0.0 to 1.0)")
    wakeword_rolling_window_size: int = Field(default=5, description="Number of prediction scores to keep in rolling window")
    wakeword_min_consecutive_detections: int = Field(default=3, description="Minimum number of consecutive frames above lower threshold to trigger")
    wakeword_cooldown_duration: float = Field(default=1.0, description="Cooldown duration in seconds after a trigger or reset")
    wakeword_lower_threshold: float = Field(default=0.2, description="Lower threshold for consecutive detection checks")
    listen_timeout: float = Field(default=8.0, description="Active listening duration in seconds")
    post_response_timeout: float = Field(default=5.0, description="Continuous follow-up listening window duration in seconds")
    voice_cooldown_seconds: float = Field(default=1.0, description="Post-TTS cooldown pause in seconds")
    voice_mode: bool = Field(default=False, description="Default to hands-free voice interface mode on startup")
    auto_install_dependencies: bool = Field(default=False, description="Automatically install missing voice dependencies on startup")

    # Memory Configuration
    memory_db_path: str = Field(default="instance/jarvis_memory.db", description="SQLite database path for memories")

    # Voice Loop FSM Configuration
    voice_speech_start_timeout: float = Field(default=8.0, description="Duration in seconds to wait for user to start speaking")
    voice_silence_timeout: float = Field(default=2.0, description="Silence duration in seconds before stopping recording")
    voice_silence_threshold: float = Field(default=500.0, description="RMS amplitude threshold below which audio is considered silent")
    voice_beep_enabled: bool = Field(default=True, description="Enable beep sound acknowledgement when wake word is detected")
    voice_beep_frequency: int = Field(default=1000, description="Acknowledgement beep sound frequency in Hz")
    voice_beep_duration: int = Field(default=200, description="Acknowledgement beep sound duration in milliseconds")
    voice_follow_up_timeout: float = Field(default=15.0, description="Duration in seconds to wait for follow-up speech")
    voice_barge_in_enabled: bool = Field(default=False, description="Enable interruption/barge-in mode during speech output")
    voice_pre_roll_buffer_ms: int = Field(default=500, description="Pre-roll audio buffer size in milliseconds")
    voice_vad_threshold: float = Field(default=0.5, description="Silero VAD threshold for voice activity detection")
    voice_summary_threshold: int = Field(default=100, description="Word count threshold above which responses are summarized for TTS")

    # STT Confidence Gate — controls when transcripts are accepted, questioned, or rejected
    stt_confidence_high: float = Field(
        default=0.75,
        description="Confidence >= this value: process normally. Range 0.0–1.0.",
    )
    stt_confidence_low: float = Field(
        default=0.50,
        description="Confidence >= this but < stt_confidence_high: ask for confirmation. Below this: reject and ask to repeat.",
    )



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
