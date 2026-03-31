"""
core/orchestrator.py – Central pipeline orchestrator.

Implements a LangGraph-inspired state-machine approach:

  ┌─────────┐      ┌────────────┐      ┌──────────┐      ┌───────────┐
  │ Planner │ ───▶ │ Researcher │ ───▶ │ Executor │ ───▶ │ Reflector │
  └─────────┘      └────────────┘      └──────────┘      └───────────┘
        │                                                       │
        └───────────── repeated for every step ────────────────┘

State is maintained in an ``OrchestratorState`` dataclass and persisted to
the SQLite memory store after every step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.reflector import ReflectorAgent
from agents.researcher import ResearcherAgent
from core.memory import (
    create_task,
    get_step_results,
    init_db,
    save_step_result,
    store_knowledge,
    update_task,
)

logger = logging.getLogger(__name__)

MAX_STEP_RETRIES = 2


def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class StepState:
    index: int
    text: str
    research: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    reflection: dict[str, Any] | None = None
    status: str = "pending"
    retries: int = 0


@dataclass
class OrchestratorState:
    goal: str
    task_id: int | None = None
    plan: dict[str, Any] | None = None
    steps: list[StepState] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "pending"  # pending | planning | running | completed | failed
    final_output: dict[str, Any] | None = None
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Drives the full Planner → Researcher → Executor → Reflector pipeline."""

    def __init__(
        self,
        model: str = "mistral",
        enable_reflection: bool = True,
        interactive: bool = False,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Ollama model name to use for all agents.
        enable_reflection:
            Whether to run the Reflector agent after each Executor output.
        interactive:
            If True, print the plan and pause for user approval before executing.
        progress_callback:
            Optional callable ``(event_name, data)`` invoked after each
            pipeline stage.  Useful for streaming progress to API clients.
        """
        self.model = model
        self.enable_reflection = enable_reflection
        self.interactive = interactive
        self.progress_callback = progress_callback

        self.planner = PlannerAgent(model=model)
        self.researcher = ResearcherAgent(model=model)
        self.executor = ExecutorAgent(model=model)
        self.reflector = ReflectorAgent(model=model) if enable_reflection else None

        init_db()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, goal: str) -> dict[str, Any]:
        """Run the full pipeline for a goal and return the structured output.

        Parameters
        ----------
        goal:
            The user's high-level request.

        Returns
        -------
        dict
            Complete execution result including plan, step outputs, and summary.
        """
        state = OrchestratorState(goal=goal)

        # Persist task to DB
        state.task_id = create_task(goal)
        update_task(state.task_id, status="planning")

        try:
            # ── Stage 1: Planning ──────────────────────────────────────────
            state.status = "planning"
            self._emit("planning_start", {"goal": goal})
            state.plan = self.planner.run(goal)
            state.steps = [
                StepState(index=i, text=step)
                for i, step in enumerate(state.plan["steps"])
            ]
            update_task(state.task_id, plan=state.plan, status="running")
            self._emit("planning_done", {"plan": state.plan})

            # ── Interactive approval ───────────────────────────────────────
            if self.interactive:
                self._interactive_approve(state.plan)

            # ── Stage 2: Per-step loop ─────────────────────────────────────
            state.status = "running"
            for step_state in state.steps:
                state.current_step_index = step_state.index
                self._run_step(state, step_state)

            # ── Stage 3: Aggregate final output ───────────────────────────
            state.final_output = self._aggregate(state)
            state.status = "completed"
            update_task(
                state.task_id,
                status="completed",
                final_output=state.final_output,
            )
            self._emit("completed", {"final_output": state.final_output})

        except Exception as exc:
            logger.exception("[Orchestrator] Pipeline failed: %s", exc)
            state.status = "failed"
            if state.task_id:
                update_task(state.task_id, status="failed")
            self._emit("failed", {"error": str(exc)})
            raise

        finally:
            state.elapsed_seconds = round(time.time() - state.started_at, 2)

        return self._serialise(state)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_step(self, state: OrchestratorState, step_state: StepState) -> None:
        goal = state.goal
        step_text = step_state.text
        step_num = step_state.index + 1
        total = len(state.steps)

        logger.info(
            "[Orchestrator] Step %d/%d: %s", step_num, total, step_text[:80]
        )
        self._emit(
            "step_start",
            {"step_index": step_state.index, "step": step_text, "total": total},
        )

        # ── Research ──────────────────────────────────────────────────────
        for attempt in range(MAX_STEP_RETRIES + 1):
            try:
                step_state.research = self.researcher.run(step_text, goal=goal)
                break
            except Exception as exc:
                if _is_timeout_error(exc):
                    logger.warning("[Orchestrator] Research timed out; skipping retries for this step")
                    step_state.research = {"step": step_text, "details": "Research unavailable (timeout).", "resources": []}
                    break
                if attempt == MAX_STEP_RETRIES:
                    logger.error("[Orchestrator] Research failed after %d attempts: %s", attempt + 1, exc)
                    step_state.research = {"step": step_text, "details": "Research unavailable.", "resources": []}
                else:
                    logger.warning("[Orchestrator] Research attempt %d failed: %s – retrying", attempt + 1, exc)

        self._emit("research_done", {"step_index": step_state.index, "research": step_state.research})

        # ── Execution ────────────────────────────────────────────────────
        for attempt in range(MAX_STEP_RETRIES + 1):
            try:
                step_state.execution = self.executor.run(
                    step_text, goal=goal, research=step_state.research
                )
                step_state.retries = attempt
                break
            except Exception as exc:
                if _is_timeout_error(exc):
                    logger.warning("[Orchestrator] Execution timed out; skipping retries for this step")
                    step_state.execution = {
                        "step": step_text,
                        "result": "Execution timed out. Try a lighter model, disable reflection, or increase OLLAMA_READ_TIMEOUT_S.",
                        "status": "failed",
                    }
                    step_state.status = "failed"
                    break
                if attempt == MAX_STEP_RETRIES:
                    logger.error("[Orchestrator] Execution failed after %d attempts: %s", attempt + 1, exc)
                    step_state.execution = {
                        "step": step_text,
                        "result": f"Execution failed: {exc}",
                        "status": "failed",
                    }
                    step_state.status = "failed"
                else:
                    logger.warning("[Orchestrator] Execution attempt %d failed: %s – retrying", attempt + 1, exc)

        self._emit("execution_done", {"step_index": step_state.index, "execution": step_state.execution})

        # ── Reflection ───────────────────────────────────────────────────
        if self.reflector and step_state.status != "failed":
            try:
                step_state.reflection = self.reflector.run(
                    step_text, step_state.execution, goal=goal
                )
                # If reflector improved the result, use the improved version
                if (
                    step_state.reflection.get("action_taken") == "improved"
                    and step_state.reflection.get("improved_result")
                ):
                    step_state.execution["result"] = step_state.reflection["improved_result"]
                    logger.info("[Orchestrator] Reflector improved step %d output", step_num)
            except Exception as exc:
                logger.warning("[Orchestrator] Reflection failed (non-fatal): %s", exc)

            self._emit("reflection_done", {"step_index": step_state.index, "reflection": step_state.reflection})

        if step_state.status != "failed":
            step_state.status = "completed"

        # ── Persist ──────────────────────────────────────────────────────
        save_step_result(
            task_id=state.task_id,
            step_index=step_state.index,
            step_text=step_text,
            research=step_state.research,
            execution=step_state.execution,
            reflection=step_state.reflection,
            status=step_state.status,
        )

        # Store useful knowledge for future tasks
        if step_state.research and step_state.research.get("details"):
            store_knowledge(
                keyword=step_text[:80],
                content=step_state.research["details"][:500],
                source_task=state.task_id,
            )

        self._emit(
            "step_done",
            {
                "step_index": step_state.index,
                "status": step_state.status,
                "retries": step_state.retries,
            },
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _aggregate(self, state: OrchestratorState) -> dict[str, Any]:
        """Combine all step outputs into a final summary."""
        step_outputs = []
        for s in state.steps:
            entry: dict[str, Any] = {
                "step_index": s.index,
                "step": s.text,
                "status": s.status,
            }
            if s.execution:
                entry["result"] = s.execution.get("result", "")
                entry["code"] = s.execution.get("code")
                entry["confidence"] = s.execution.get("confidence", 0.7)
            if s.reflection:
                entry["quality_score"] = s.reflection.get("quality_score")
            step_outputs.append(entry)

        completed = sum(1 for s in state.steps if s.status == "completed")
        return {
            "goal": state.goal,
            "total_steps": len(state.steps),
            "completed_steps": completed,
            "failed_steps": len(state.steps) - completed,
            "step_outputs": step_outputs,
            "overall_status": "completed" if completed == len(state.steps) else "partial",
        }

    def _serialise(self, state: OrchestratorState) -> dict[str, Any]:
        return {
            "task_id": state.task_id,
            "goal": state.goal,
            "status": state.status,
            "plan": state.plan,
            "final_output": state.final_output,
            "elapsed_seconds": state.elapsed_seconds,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Fire the progress callback if registered."""
        if self.progress_callback:
            try:
                self.progress_callback(event, data)
            except Exception as exc:
                logger.debug("Progress callback raised: %s", exc)

    @staticmethod
    def _interactive_approve(plan: dict[str, Any]) -> None:
        """Print plan and wait for user confirmation."""
        print("\n" + "=" * 60)
        print(f"GOAL: {plan['goal']}")
        print(f"COMPLEXITY: {plan.get('estimated_complexity', 'unknown')}")
        print("STEPS:")
        for i, step in enumerate(plan.get("steps", []), 1):
            print(f"  {i}. {step}")
        print("=" * 60)
        answer = input("\nApprove this plan? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("User rejected the plan.")
