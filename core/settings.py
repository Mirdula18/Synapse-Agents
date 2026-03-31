"""
core/settings.py - Runtime configuration from environment variables.

Keeps deployment controls (auth, CORS, Ollama behavior) in one place so
the API and LLM layers do not rely on hardcoded defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    cors_origins: list[str]
    default_model: str
    ollama_base_url: str
    ollama_connect_timeout_s: int
    ollama_read_timeout_s: int
    ollama_retries: int
    ollama_retry_backoff_s: float
    ollama_num_predict: int
    ollama_temperature: float
    api_history_default_limit: int


def load_settings() -> Settings:
    raw_origins = os.getenv("SYNAPSE_CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    return Settings(
        api_key=os.getenv("SYNAPSE_API_KEY"),
        cors_origins=origins or ["*"],
        default_model=os.getenv("SYNAPSE_DEFAULT_MODEL", "mistral"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_connect_timeout_s=_as_int(os.getenv("OLLAMA_CONNECT_TIMEOUT_S"), 10, minimum=1),
        ollama_read_timeout_s=_as_int(os.getenv("OLLAMA_READ_TIMEOUT_S"), 45, minimum=5),
        ollama_retries=_as_int(os.getenv("OLLAMA_RETRIES"), 2, minimum=1),
        ollama_retry_backoff_s=float(os.getenv("OLLAMA_RETRY_BACKOFF_S", "1.5")),
        ollama_num_predict=_as_int(os.getenv("OLLAMA_NUM_PREDICT"), 700, minimum=64),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        api_history_default_limit=_as_int(os.getenv("SYNAPSE_HISTORY_LIMIT"), 20, minimum=1),
    )


SETTINGS = load_settings()
