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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"
REQUEST_TIMEOUT = 120  # seconds

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
    retries: int = 3,
    backoff: float = 2.0,
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
    system_instruction = SYSTEM_ROLES.get(system_role, SYSTEM_ROLES["executor"])
    full_prompt = f"[SYSTEM]\n{system_instruction}\n\n[USER]\n{prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
    }

    last_error: Exception | None = None
    wait = backoff

    for attempt in range(1, retries + 1):
        try:
            logger.debug("LLM request attempt %d/%d (role=%s)", attempt, retries, system_role)
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            raw = response.json()
            text = raw.get("response", "")
            parsed = _extract_json(text)
            parsed.setdefault("confidence", _estimate_confidence(text))
            logger.debug("LLM response received (role=%s)", system_role)
            return parsed

        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "LLM attempt %d failed: %s – retrying in %.1fs", attempt, exc, wait
            )
            time.sleep(wait)
            wait *= 2

    raise RuntimeError(
        f"All {retries} LLM attempts failed. Last error: {last_error}"
    )


def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Return True when the Ollama server is reachable and the model is loaded."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        return any(t.get("name", "").startswith(model) for t in tags)
    except Exception:
        return False


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


def _estimate_confidence(text: str) -> float:
    """Heuristic confidence score based on response length/completeness."""
    length = len(text.strip())
    if length < 50:
        return 0.3
    if length < 200:
        return 0.6
    return min(0.95, 0.7 + length / 5000)
