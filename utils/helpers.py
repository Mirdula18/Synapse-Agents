"""
utils/helpers.py – Shared utility functions.

Provides logging configuration, JSON serialisation helpers, safe shell
execution, file reading, and a simple code-extraction utility.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a human-friendly format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Serialise *obj* to JSON, falling back to str() for non-serialisable types."""
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(obj), indent=indent, ensure_ascii=False)


def truncate(text: str, max_chars: int = 500) -> str:
    """Return *text* truncated to *max_chars* with a marker if clipped."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[truncated]"


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def read_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read and return a text file's content.

    Returns an error message string (never raises) so agents can handle it.
    """
    try:
        return Path(path).read_text(encoding=encoding)
    except FileNotFoundError:
        return f"[ERROR] File not found: {path}"
    except PermissionError:
        return f"[ERROR] Permission denied: {path}"
    except Exception as exc:
        return f"[ERROR] Could not read {path}: {exc}"


def write_file(path: str | Path, content: str, encoding: str = "utf-8") -> bool:
    """Write *content* to *path*, creating parent directories as needed.

    Returns True on success, False on failure.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return True
    except Exception as exc:
        logger.error("write_file failed for %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extract fenced code blocks from markdown text.

    Returns a list of ``{"language": ..., "code": ...}`` dicts.
    """
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(text):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        blocks.append({"language": lang, "code": code})
    return blocks


# ---------------------------------------------------------------------------
# Safe shell execution
# ---------------------------------------------------------------------------

_ALLOWED_COMMANDS: set[str] = {
    "echo",
    "ls",
    "pwd",
    "cat",
    "python",
    "python3",
    "pip",
    "pip3",
    "node",
    "npm",
    "git",
}


def safe_shell(
    command: list[str],
    timeout: int = 30,
    allowed_commands: set[str] | None = None,
) -> dict[str, Any]:
    """Execute a shell command in safe mode.

    Only commands whose base name appears in *allowed_commands* (or the
    built-in whitelist) are permitted.  Returns a dict with stdout/stderr/
    return_code keys.
    """
    allowed = allowed_commands or _ALLOWED_COMMANDS
    if not command:
        return {"stdout": "", "stderr": "Empty command", "return_code": 1}

    base = Path(command[0]).name
    if base not in allowed:
        msg = f"Command '{base}' is not in the safe-command whitelist."
        logger.warning("safe_shell blocked: %s", msg)
        return {"stdout": "", "stderr": msg, "return_code": 1}

    try:
        if os.name == "nt" and base == "echo":
            result = subprocess.run(
                ["cmd", "/c", *command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "return_code": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "return_code": -1}


# ---------------------------------------------------------------------------
# Plugin loader
# ---------------------------------------------------------------------------


_EXPECTED_MODULE_PARTS = 2


def load_agent_plugin(module_path: str) -> Any:
    """Dynamically load an agent class from a dotted module path.

    Example: ``load_agent_plugin("plugins.my_agent.MyAgent")``
    """
    import importlib

    parts = module_path.rsplit(".", 1)
    if len(parts) != _EXPECTED_MODULE_PARTS:
        raise ValueError(f"module_path must be 'module.ClassName', got: {module_path!r}")
    module_name, class_name = parts
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    logger.info("Loaded plugin agent: %s", module_path)
    return cls
