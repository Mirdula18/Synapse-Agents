"""
agents/reflector.py – Self-Reflection Agent.

Responsibility: Review the executor's output for a step, identify issues,
and if quality is below threshold, generate an improved version.

Output schema:
{
    "step": "<step text>",
    "original_result": "<the executor's result>",
    "issues_found": ["issue 1", ...],
    "improved_result": "<improved version or null if original was good>",
    "quality_score": 0.0-1.0,
    "action_taken": "accepted|improved",
    "confidence": 0.0-1.0
}
"""

from __future__ import annotations

import logging
from typing import Any

from core.llm import generate_response

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.75  # Trigger improvement if quality_score < this

REFLECTOR_PROMPT_TEMPLATE = """
You are reviewing the following task execution output for quality and correctness.

STEP: {step}
OVERALL GOAL: {goal}

EXECUTOR OUTPUT:
{executor_output}

Evaluate the output:
1. Is it complete and correct?
2. Does it fully address the step?
3. Are there bugs, gaps, or improvements needed?

Return ONLY a JSON object in exactly this format:
{{
  "step": "<restate the step>",
  "original_result": "<summarise the executor output>",
  "issues_found": ["issue 1", "issue 2"],
  "improved_result": "<improved/corrected version if needed, otherwise null>",
  "quality_score": <float 0.0 to 1.0>,
  "action_taken": "<accepted|improved>"
}}
""".strip()


class ReflectorAgent:
    """Reviews and optionally improves executor outputs."""

    def __init__(
        self,
        model: str = "mistral",
        quality_threshold: float = QUALITY_THRESHOLD,
    ) -> None:
        self.model = model
        self.role = "reflector"
        self.quality_threshold = quality_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        step: str,
        execution_result: dict[str, Any],
        goal: str = "",
    ) -> dict[str, Any]:
        """Reflect on an execution result and return a (possibly improved) output.

        Parameters
        ----------
        step:
            The step text.
        execution_result:
            The executor agent's output dict.
        goal:
            The parent goal for broader context.

        Returns
        -------
        dict
            Reflection result (see module docstring for schema).
        """
        logger.info("[Reflector] Reviewing step: %s", step[:100])

        executor_output = execution_result.get("result", "")
        prompt = REFLECTOR_PROMPT_TEMPLATE.format(
            step=step,
            goal=goal or "Not specified",
            executor_output=executor_output[:3000],  # Guard against token limits
        )

        raw = generate_response(prompt, system_role=self.role, model=self.model)
        result = self._validate(raw, step, executor_output)

        quality = result.get("quality_score", 1.0)
        logger.info(
            "[Reflector] Quality score=%.2f, action=%s",
            quality,
            result.get("action_taken"),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(
        self, raw: dict[str, Any], original_step: str, original_output: str
    ) -> dict[str, Any]:
        step = raw.get("step") or original_step
        quality_score = float(raw.get("quality_score", 1.0))
        quality_score = max(0.0, min(1.0, quality_score))

        issues = raw.get("issues_found", [])
        improved = raw.get("improved_result")
        action = raw.get("action_taken", "accepted")

        # Sanity check: if no issues and quality is high, mark as accepted
        if quality_score >= self.quality_threshold and not issues:
            action = "accepted"
            improved = None

        return {
            "step": step,
            "original_result": raw.get("original_result", original_output[:500]),
            "issues_found": [str(i) for i in issues] if isinstance(issues, list) else [],
            "improved_result": improved,
            "quality_score": quality_score,
            "action_taken": action if action in {"accepted", "improved"} else "accepted",
            "confidence": raw.get("confidence", quality_score),
        }
