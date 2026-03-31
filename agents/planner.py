"""
agents/planner.py – Planner Agent.

Responsibility: Accept a high-level user goal and decompose it into an
ordered list of concrete, executable steps with an estimated complexity.

Output schema:
{
    "goal": "<original goal>",
    "steps": ["step 1", "step 2", ...],
    "estimated_complexity": "low|medium|high",
    "confidence": 0.0-1.0
}
"""

from __future__ import annotations

import logging
from typing import Any

from core.llm import generate_response

logger = logging.getLogger(__name__)

PLANNER_PROMPT_TEMPLATE = """
You are planning the execution of the following goal:

GOAL: {goal}

Break this goal down into a clear, ordered list of concrete steps that a software
development team would follow.  Each step should be actionable and specific.

Return ONLY a JSON object in exactly this format (no extra keys, no prose):
{{
  "goal": "<restate the goal concisely>",
  "steps": [
    "Step 1 description",
    "Step 2 description"
  ],
  "estimated_complexity": "<low|medium|high>"
}}
""".strip()


class PlannerAgent:
    """Decomposes a user goal into an ordered execution plan."""

    def __init__(self, model: str = "mistral") -> None:
        self.model = model
        self.role = "planner"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, goal: str) -> dict[str, Any]:
        """Generate and return a structured plan for the given goal.

        Parameters
        ----------
        goal:
            The high-level user request.

        Returns
        -------
        dict
            Validated plan dictionary (see module docstring for schema).
        """
        logger.info("[Planner] Planning goal: %s", goal[:120])
        prompt = PLANNER_PROMPT_TEMPLATE.format(goal=goal)
        raw = generate_response(prompt, system_role=self.role, model=self.model)
        plan = self._validate(raw, goal)
        logger.info(
            "[Planner] Plan ready – %d steps (complexity=%s, confidence=%.2f)",
            len(plan["steps"]),
            plan["estimated_complexity"],
            plan.get("confidence", 0),
        )
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, raw: dict[str, Any], original_goal: str) -> dict[str, Any]:
        """Ensure required fields are present and sensible."""
        goal = raw.get("goal") or original_goal
        steps = raw.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            logger.warning("[Planner] LLM returned no steps; using fallback single step")
            steps = [f"Complete the following goal: {original_goal}"]
        complexity = raw.get("estimated_complexity", "medium").lower()
        if complexity not in {"low", "medium", "high"}:
            complexity = "medium"
        return {
            "goal": goal,
            "steps": [str(s) for s in steps],
            "estimated_complexity": complexity,
            "confidence": raw.get("confidence", 0.7),
        }
