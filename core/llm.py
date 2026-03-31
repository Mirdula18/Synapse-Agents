"""
core/llm.py – Reusable Ollama LLM interface.

All agents call generate_response() to interact with the local Ollama server
(http://localhost:11434).  No internet access is required at runtime.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from requests import Timeout

from core.settings import SETTINGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = f"{SETTINGS.ollama_base_url}/api/generate"
OLLAMA_TAGS_URL = f"{SETTINGS.ollama_base_url}/api/tags"
DEFAULT_MODEL = SETTINGS.default_model
CONNECT_TIMEOUT = SETTINGS.ollama_connect_timeout_s
READ_TIMEOUT = SETTINGS.ollama_read_timeout_s
MIN_NUM_PREDICT = 128
ROLE_NUM_PREDICT: dict[str, int] = {
    "planner": 192,
    "researcher": 420,
    "executor": SETTINGS.ollama_num_predict,
    "reflector": 220,
}

SYSTEM_ROLES: dict[str, str] = {
    "planner": (
        "You are a senior software architect and project planner. "
        "Your job is to break down a user's goal into a clear, ordered list of steps. "
        "Always respond with valid JSON only – no prose, no markdown fences."
    ),
    "researcher": (
        "You are a thorough technical researcher. "
        "Given a single task step, you provide detailed context, best practices, "
        "relevant libraries, and any gotchas the executor needs to know. "
        "Always respond with valid JSON only – no prose, no markdown fences."
    ),
    "executor": (
        "You are an expert software engineer and code generator. "
        "You receive a task step enriched with research context and produce a "
        "concrete, working implementation or result. "
        "Always respond with valid JSON only – no prose, no markdown fences."
    ),
    "reflector": (
        "You are a critical code reviewer and quality assurance expert. "
        "You review completed task outputs, identify issues or improvements, "
        "and provide an improved version when needed. "
        "Always respond with valid JSON only – no prose, no markdown fences."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_response(
    prompt: str,
    system_role: str = "executor",
    model: str = DEFAULT_MODEL,
    retries: int | None = None,
    backoff: float | None = None,
) -> dict[str, Any]:
    """Send a prompt to Ollama and return a parsed JSON dict.

    Parameters
    ----------
    prompt:
        The user-facing prompt text.
    system_role:
        One of ``"planner"``, ``"researcher"``, ``"executor"``, ``"reflector"``.
        Selects the appropriate system instruction.
    model:
        Ollama model name (e.g. ``"mistral"``, ``"llama3"``).
    retries:
        Number of retry attempts on network/parse failure.
    backoff:
        Seconds to wait between retries (doubles each attempt).

    Returns
    -------
    dict
        Parsed JSON response from the LLM.

    Raises
    ------
    RuntimeError
        When all retry attempts are exhausted.
    """
    retries = retries if retries is not None else SETTINGS.ollama_retries
    backoff = backoff if backoff is not None else SETTINGS.ollama_retry_backoff_s

    system_instruction = SYSTEM_ROLES.get(system_role, SYSTEM_ROLES["executor"])
    full_prompt = f"[SYSTEM]\n{system_instruction}\n\n[USER]\n{prompt}"
    start_num_predict = ROLE_NUM_PREDICT.get(system_role, SETTINGS.ollama_num_predict)
    current_num_predict = max(MIN_NUM_PREDICT, start_num_predict)

    last_error: Exception | None = None
    wait = backoff

    for attempt in range(1, retries + 1):
        try:
            logger.debug("LLM request attempt %d/%d (role=%s)", attempt, retries, system_role)
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": SETTINGS.ollama_temperature,
                    "num_predict": current_num_predict,
                },
            }
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            response.raise_for_status()

            text = _extract_response_text(response)
            parsed = _extract_json(text)
            parsed.setdefault("confidence", _estimate_confidence(text))
            logger.debug("LLM response received (role=%s)", system_role)
            return parsed

        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, Timeout):
                logger.warning(
                    "LLM timeout (connect=%ss, read=%ss, num_predict=%s) on attempt %d/%d",
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                    current_num_predict,
                    attempt,
                    retries,
                )
                # If the model is slow, request fewer tokens on the next attempt.
                current_num_predict = max(MIN_NUM_PREDICT, int(current_num_predict * 0.7))
            if attempt < retries:
                logger.warning(
                    "LLM attempt %d failed: %s – retrying in %.1fs", attempt, exc, wait
                )
                time.sleep(wait)
                wait *= 2
            else:
                logger.error("LLM attempt %d failed: %s – no retries left", attempt, exc)

    raise RuntimeError(
        f"All {retries} LLM attempts failed. Last error: {last_error}"
    )


def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Return True when the Ollama server is reachable and the model is loaded."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=(CONNECT_TIMEOUT, 5))
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        return any(t.get("name", "").startswith(model) for t in tags)
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return the installed Ollama model names from /api/tags."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=(CONNECT_TIMEOUT, 8))
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names = [m.get("name", "") for m in models if m.get("name")]
        # Normalise to base names first, then keep full tags if duplicates exist.
        base_names = [n.split(":", 1)[0] for n in names]
        unique = sorted(set(base_names or names))
        return unique
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling optional markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")


def _extract_response_text(response: requests.Response) -> str:
    """Extract the generated text from Ollama responses with tolerant parsing.

    Some Ollama setups may return newline-delimited JSON chunks even when
    ``stream`` is false. This helper handles both a single JSON object and
    NDJSON chunk payloads.
    """
    try:
        raw = response.json()
    except ValueError:
        raw = _parse_json_or_ndjson(getattr(response, "text", ""))

    if isinstance(raw, dict):
        text = raw.get("response", "")
        if isinstance(text, str):
            return text
    raise ValueError("Ollama response did not include a valid 'response' string")


def _parse_json_or_ndjson(text: str) -> dict[str, Any]:
    """Parse either a JSON object or NDJSON payload from Ollama."""
    payload = (text or "").strip()
    if not payload:
        raise ValueError("Empty response body from Ollama")

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    chunks: list[str] = []
    last_obj: dict[str, Any] | None = None
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last_obj = obj
            part = obj.get("response")
            if isinstance(part, str):
                chunks.append(part)

    if chunks:
        return {"response": "".join(chunks)}
    if last_obj is not None:
        return last_obj
    raise ValueError(f"Unable to parse Ollama payload: {payload[:200]!r}")


def _estimate_confidence(text: str) -> float:
    """Heuristic confidence score based on response length/completeness."""
    length = len(text.strip())
    if length < 50:
        return 0.3
    if length < 200:
        return 0.6
    return min(0.95, 0.7 + length / 5000)
