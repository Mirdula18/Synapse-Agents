"""
api/routes.py – FastAPI router.

Endpoints:
  POST /run-task     – Execute a goal through the full agent pipeline.
  GET  /history      – List past tasks.
  GET  /task/{id}    – Get details of a specific task.
  GET  /health       – Liveness check (also verifies Ollama connectivity).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from core.llm import is_ollama_available
from core.memory import get_step_results, get_task, init_db, list_tasks
from core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

init_db()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RunTaskRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=2000, description="The user's high-level goal")
    model: str = Field("mistral", description="Ollama model name to use")
    enable_reflection: bool = Field(True, description="Run reflector agent after each step")
    interactive: bool = Field(
        False,
        description="If true, return the plan first and require a separate /approve call",
    )


class RunTaskResponse(BaseModel):
    task_id: int
    goal: str
    status: str
    plan: dict[str, Any] | None
    final_output: dict[str, Any] | None
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(model: str = Query("mistral", description="Ollama model to probe")) -> HealthResponse:
    """Return service liveness and Ollama connectivity status."""
    available = is_ollama_available(model)
    return HealthResponse(
        status="ok",
        ollama_available=available,
        model=model,
    )


@router.post("/run-task", response_model=RunTaskResponse, tags=["Tasks"])
def run_task(request: RunTaskRequest) -> RunTaskResponse:
    """Execute a user goal through the full Planner→Researcher→Executor→Reflector pipeline.

    This call is **synchronous** – it blocks until the pipeline completes.
    For long tasks consider the async variant (future work).
    """
    logger.info("POST /run-task – goal=%s", request.goal[:80])

    events: list[dict[str, Any]] = []

    def on_progress(event: str, data: dict[str, Any]) -> None:
        events.append({"event": event, **data})
        logger.info("[Pipeline] %s", event)

    try:
        orchestrator = Orchestrator(
            model=request.model,
            enable_reflection=request.enable_reflection,
            interactive=False,  # API mode – never block on stdin
            progress_callback=on_progress,
        )
        result = orchestrator.run(request.goal)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunTaskResponse(**result)


@router.get("/history", tags=["Tasks"])
def get_history(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
    """Return the most recent *limit* tasks from history."""
    return list_tasks(limit=limit)


@router.get("/task/{task_id}", tags=["Tasks"])
def get_task_detail(task_id: int) -> dict[str, Any]:
    """Return full details for a single task including all step results."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task["step_results"] = get_step_results(task_id)
    return task
