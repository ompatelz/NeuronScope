"""Configuration loaded from the environment at process startup."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the API and future training services."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEURONSCOPE_")

    app_name: str = "NeuronScope API"
    api_v1_prefix: str = "/api/v1"
    frontend_dist: Path | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
