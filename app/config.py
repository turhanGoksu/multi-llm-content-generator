"""Application settings, loaded once from environment variables / .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of every environment variable the app reads.

    Model names have no defaults on purpose: .env is their single source of
    truth, so a deprecated model is fixed by editing .env, not code.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: SecretStr
    groq_api_key: SecretStr
    gemini_model: str
    groq_model: str
    # Optional; only for Groq reasoning models (e.g. "low"). Empty = not sent.
    groq_reasoning_effort: str | None = None
    provider_timeout_seconds: float = 30.0
    database_url: str


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (read from env on first call)."""
    return Settings()
