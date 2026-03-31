"""
agents/researcher.py – Researcher Agent.

Responsibility: Given a single plan step, gather detailed context, suggest
relevant libraries/tools, note best practices and potential pitfalls.

Output schema per step:
{
    "step": "<the step text>",
    "details": "<comprehensive explanation>",
    "resources": ["resource 1", "resource 2", ...],
    "best_practices": ["...", ...],
    "confidence": 0.0-1.0
}
"""

from __future__ import annotations

import logging
from typing import Any

from core.llm import generate_response
from core.memory import search_knowledge

logger = logging.getLogger(__name__)

RESEARCHER_PROMPT_TEMPLATE = """
You are researching how to implement the following task step:

STEP: {step}

OVERALL GOAL CONTEXT: {goal}

{knowledge_context}

Provide:
1. A thorough explanation of how to accomplish this step.
2. Specific libraries, tools, or APIs to use.
3. Best practices to follow.
4. Common pitfalls to avoid.

Return ONLY a JSON object in exactly this format (no extra keys, no prose):
{{
  "step": "<restate the step>",
  "details": "<comprehensive explanation>",
  "resources": ["library/tool/api 1", "library/tool/api 2"],
  "best_practices": ["practice 1", "practice 2"],
  "pitfalls": ["pitfall 1", "pitfall 2"]
}}
""".strip()


class ResearcherAgent:
    """Enriches each plan step with relevant technical context."""

    def __init__(self, model: str = "mistral") -> None:
        self.model = model
        self.role = "researcher"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, step: str, goal: str = "") -> dict[str, Any]:
        """Research a single plan step and return enriched context.

        Parameters
        ----------
        step:
            The step text from the planner.
        goal:
            The parent goal for broader context.

        Returns
        -------
        dict
            Research context (see module docstring for schema).
        """
        logger.info("[Researcher] Researching step: %s", step[:100])

        # Pull any relevant prior knowledge from memory
        knowledge_snippets = search_knowledge(step)
        knowledge_context = self._format_knowledge(knowledge_snippets)

        prompt = RESEARCHER_PROMPT_TEMPLATE.format(
            step=step,
            goal=goal or "Not specified",
            knowledge_context=knowledge_context,
        )
        raw = generate_response(prompt, system_role=self.role, model=self.model)
        result = self._validate(raw, step)
        logger.info(
            "[Researcher] Research complete for step (confidence=%.2f)",
            result.get("confidence", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, raw: dict[str, Any], original_step: str) -> dict[str, Any]:
        step = raw.get("step") or original_step
        details = raw.get("details", "No details provided.")
        resources = raw.get("resources", [])
        best_practices = raw.get("best_practices", [])
        pitfalls = raw.get("pitfalls", [])
        return {
            "step": step,
            "details": str(details),
            "resources": [str(r) for r in resources] if isinstance(resources, list) else [],
            "best_practices": (
                [str(p) for p in best_practices] if isinstance(best_practices, list) else []
            ),
            "pitfalls": [str(p) for p in pitfalls] if isinstance(pitfalls, list) else [],
            "confidence": raw.get("confidence", 0.7),
        }

    @staticmethod
    def _format_knowledge(snippets: list[dict[str, Any]]) -> str:
        if not snippets:
            return ""
        lines = ["RELEVANT PRIOR KNOWLEDGE (from memory):"]
        for s in snippets:
            lines.append(f"- [{s.get('keyword', '')}]: {s.get('content', '')[:200]}")
        return "\n".join(lines) + "\n"
