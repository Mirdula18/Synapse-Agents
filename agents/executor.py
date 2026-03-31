"""
agents/executor.py – Executor Agent.

Responsibility: Given an enriched step (plan step + research context),
produce a concrete implementation: code, configuration, explanations, etc.

Output schema:
{
    "step": "<the step text>",
    "result": "<full implementation or answer>",
    "code": "<code block if applicable, else null>",
    "explanation": "<why this approach was chosen>",
    "status": "completed|failed",
    "confidence": 0.0-1.0
}
"""

from __future__ import annotations

import logging
from typing import Any

from core.llm import generate_response

logger = logging.getLogger(__name__)

EXECUTOR_PROMPT_TEMPLATE = """
You are implementing the following task step.

STEP: {step}

OVERALL GOAL: {goal}

RESEARCH CONTEXT:
{research_context}

Produce the final implementation, code, or answer for this step.
Be thorough and complete – do not leave placeholders.

Return ONLY a JSON object in exactly this format (no extra keys, no prose):
{{
  "step": "<restate the step>",
  "result": "<full implementation, code, configuration, or answer>",
  "code": "<standalone code block if applicable, otherwise null>",
  "explanation": "<brief rationale for your approach>",
  "status": "completed"
}}
""".strip()


class ExecutorAgent:
    """Executes each enriched step and produces a concrete result."""

    def __init__(self, model: str = "mistral") -> None:
        self.model = model
        self.role = "executor"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        step: str,
        goal: str = "",
        research: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single plan step and return the result.

        Parameters
        ----------
        step:
            The step text from the planner.
        goal:
            The parent goal for broader context.
        research:
            The researcher output for this step.

        Returns
        -------
        dict
            Execution result (see module docstring for schema).
        """
        logger.info("[Executor] Executing step: %s", step[:100])
        research_context = self._format_research(research)
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            step=step,
            goal=goal or "Not specified",
            research_context=research_context,
        )
        raw = generate_response(prompt, system_role=self.role, model=self.model)
        result = self._validate(raw, step)
        logger.info(
            "[Executor] Step completed (status=%s, confidence=%.2f)",
            result["status"],
            result.get("confidence", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, raw: dict[str, Any], original_step: str) -> dict[str, Any]:
        step = raw.get("step") or original_step
        result_text = raw.get("result", "")
        status = raw.get("status", "completed")
        if status not in {"completed", "failed"}:
            status = "completed"
        return {
            "step": step,
            "result": str(result_text) if result_text else "No output generated.",
            "code": raw.get("code"),
            "explanation": raw.get("explanation", ""),
            "status": status,
            "confidence": raw.get("confidence", 0.7),
        }

    @staticmethod
    def _format_research(research: dict[str, Any] | None) -> str:
        if not research:
            return "No research context available."
        lines = []
        if research.get("details"):
            lines.append(f"Details: {research['details']}")
        if research.get("resources"):
            lines.append("Resources: " + ", ".join(research["resources"]))
        if research.get("best_practices"):
            lines.append("Best practices: " + "; ".join(research["best_practices"]))
        if research.get("pitfalls"):
            lines.append("Pitfalls to avoid: " + "; ".join(research["pitfalls"]))
        return "\n".join(lines) if lines else "No research context available."
