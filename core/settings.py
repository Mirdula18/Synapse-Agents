"""
core/settings.py - Runtime configuration from environment variables.

Keeps deployment controls (auth, CORS, Ollama behavior) in one place so
the API and LLM layers do not rely on hardcoded defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


_LOCAL_DEV_CORS_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

logger = logging.getLogger(__name__)


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


def _as_float(
    value: str | None,
    default: float,
    *,
    var_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float for %s=%r; using default %s",
            var_name,
            value,
            default,
        )
        return default

    if minimum is not None and parsed < minimum:
        logger.warning(
            "Out-of-range float for %s=%s; clamping to minimum %s",
            var_name,
            parsed,
            minimum,
        )
        return minimum

    if maximum is not None and parsed > maximum:
        logger.warning(
            "Out-of-range float for %s=%s; clamping to maximum %s",
            var_name,
            parsed,
            maximum,
        )
        return maximum

    return parsed


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    environment: str
    cors_origins: list[str]
    cors_allow_credentials: bool
    default_model: str
    ollama_base_url: str
    ollama_connect_timeout_s: int
    ollama_read_timeout_s: int
    ollama_retries: int
    ollama_retry_backoff_s: float
    ollama_num_predict: int
    ollama_temperature: float
    api_history_default_limit: int
    async_job_retention_hours: int
    enable_exec: bool
    max_workers: int
    database_url: str | None
    kb_min_quality: float
    kb_max_entries: int


def _normalise_environment(value: str | None) -> str:
    raw = (value or "development").strip().lower()
    if raw in {"prod", "production"}:
        return "production"
    return "development"


def _parse_cors_origins(raw_origins: str | None, environment: str) -> list[str]:
    if raw_origins is None:
        if environment == "production":
            return []
        return list(_LOCAL_DEV_CORS_ORIGINS)
    return [o.strip() for o in raw_origins.split(",") if o.strip()]


def _validate_cors(origins: list[str], allow_credentials: bool) -> None:
    if allow_credentials:
        if "*" in origins:
            raise ValueError(
                "Wildcard '*' is not allowed in SYNAPSE_CORS_ORIGINS when "
                "SYNAPSE_CORS_ALLOW_CREDENTIALS is enabled."
            )
        if not origins:
            raise ValueError(
                "Explicit SYNAPSE_CORS_ORIGINS are required when "
                "SYNAPSE_CORS_ALLOW_CREDENTIALS is enabled."
            )


def load_settings() -> Settings:
    environment = _normalise_environment(os.getenv("SYNAPSE_ENV"))
    allow_credentials = _as_bool(os.getenv("SYNAPSE_CORS_ALLOW_CREDENTIALS"), True)
    origins = _parse_cors_origins(os.getenv("SYNAPSE_CORS_ORIGINS"), environment)
    _validate_cors(origins, allow_credentials)

    if not origins and not allow_credentials:
        origins = ["*"]

    return Settings(
        api_key=os.getenv("SYNAPSE_API_KEY"),
        environment=environment,
        cors_origins=origins,
        cors_allow_credentials=allow_credentials,
        default_model=os.getenv("SYNAPSE_DEFAULT_MODEL", "mistral"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_connect_timeout_s=_as_int(os.getenv("OLLAMA_CONNECT_TIMEOUT_S"), 10, minimum=1),
        ollama_read_timeout_s=_as_int(os.getenv("OLLAMA_READ_TIMEOUT_S"), 60, minimum=5),
        ollama_retries=_as_int(os.getenv("OLLAMA_RETRIES"), 1, minimum=1),
        ollama_retry_backoff_s=_as_float(
            os.getenv("OLLAMA_RETRY_BACKOFF_S"),
            1.5,
            var_name="OLLAMA_RETRY_BACKOFF_S",
            minimum=0.1,
            maximum=120.0,
        ),
        ollama_num_predict=_as_int(os.getenv("OLLAMA_NUM_PREDICT"), 520, minimum=64),
        ollama_temperature=_as_float(
            os.getenv("OLLAMA_TEMPERATURE"),
            0.2,
            var_name="OLLAMA_TEMPERATURE",
            minimum=0.0,
            maximum=2.0,
        ),
        api_history_default_limit=_as_int(os.getenv("SYNAPSE_HISTORY_LIMIT"), 20, minimum=1),
        async_job_retention_hours=_as_int(
            os.getenv("SYNAPSE_ASYNC_JOB_RETENTION_HOURS"),
            24,
            minimum=1,
        ),
        enable_exec=_as_bool(os.getenv("SYNAPSE_ENABLE_EXEC"), False),
        max_workers=_as_int(os.getenv("SYNAPSE_MAX_WORKERS"), 2, minimum=1),
        database_url=os.getenv("DATABASE_URL"),
        kb_min_quality=_as_float(
            os.getenv("SYNAPSE_KB_MIN_QUALITY"),
            0.5,
            var_name="SYNAPSE_KB_MIN_QUALITY",
            minimum=0.0,
            maximum=1.0,
        ),
        kb_max_entries=_as_int(os.getenv("SYNAPSE_KB_MAX_ENTRIES"), 1000, minimum=0),
    )


SETTINGS = load_settings()
