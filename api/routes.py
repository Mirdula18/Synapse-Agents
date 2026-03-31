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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from core.llm import is_ollama_available, list_available_models
from core.memory import get_step_results, get_task, init_db, list_tasks
from core.orchestrator import Orchestrator
from core.settings import SETTINGS

logger = logging.getLogger(__name__)
router = APIRouter()

init_db()
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synapse-job")


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


class AsyncRunTaskResponse(BaseModel):
    job_id: str
    status: str
    goal: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    goal: str
    model: str
    created_at: float
    updated_at: float
    events: list[dict[str, Any]]
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    model: str


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = SETTINGS.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _record_event(job_id: str, event: str, data: dict[str, Any]) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["events"].append({"event": event, "ts": time.time(), **data})
        job["updated_at"] = time.time()


def _run_job(job_id: str, request: RunTaskRequest) -> None:
    def on_progress(event: str, data: dict[str, Any]) -> None:
        _record_event(job_id, event, data)

    try:
        with _JOBS_LOCK:
            if job_id not in _JOBS:
                return
            _JOBS[job_id]["status"] = "running"
            _JOBS[job_id]["updated_at"] = time.time()

        orchestrator = Orchestrator(
            model=request.model,
            enable_reflection=request.enable_reflection,
            interactive=False,
            progress_callback=on_progress,
        )
        result = orchestrator.run(request.goal)
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "completed"
                _JOBS[job_id]["result"] = result
                _JOBS[job_id]["updated_at"] = time.time()
    except Exception as exc:
        logger.exception("Async pipeline error: %s", exc)
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["error"] = str(exc)
                _JOBS[job_id]["updated_at"] = time.time()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(model: str = Query(SETTINGS.default_model, description="Ollama model to probe")) -> HealthResponse:
    """Return service liveness and Ollama connectivity status."""
    available = is_ollama_available(model)
    return HealthResponse(
        status="ok",
        ollama_available=available,
        model=model,
    )


@router.get("/models", response_model=ModelsResponse, tags=["System"])
def get_models() -> ModelsResponse:
    """Return available local Ollama models for dropdown selection."""
    models = list_available_models()
    if not models:
        models = [SETTINGS.default_model]
    return ModelsResponse(models=models, default_model=SETTINGS.default_model)


@router.post("/run-task", response_model=RunTaskResponse, tags=["Tasks"])
def run_task(request: RunTaskRequest, _auth: None = Depends(_require_api_key)) -> RunTaskResponse:
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


@router.post("/run-task-async", response_model=AsyncRunTaskResponse, tags=["Tasks"])
def run_task_async(request: RunTaskRequest, _auth: None = Depends(_require_api_key)) -> AsyncRunTaskResponse:
    """Submit a task for background execution and return a job id."""
    job_id = str(uuid.uuid4())
    now = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "goal": request.goal,
            "model": request.model,
            "created_at": now,
            "updated_at": now,
            "events": [],
            "result": None,
            "error": None,
        }
    _EXECUTOR.submit(_run_job, job_id, request)
    return AsyncRunTaskResponse(job_id=job_id, status="queued", goal=request.goal)


@router.get("/run-task-async/{job_id}", response_model=JobStatusResponse, tags=["Tasks"])
def run_task_status(job_id: str, _auth: None = Depends(_require_api_key)) -> JobStatusResponse:
    """Return background job status, events, and final result when done."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return JobStatusResponse(**job)


@router.get("/history", tags=["Tasks"])
def get_history(
    limit: int = Query(SETTINGS.api_history_default_limit, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _auth: None = Depends(_require_api_key),
) -> list[dict[str, Any]]:
    """Return the most recent *limit* tasks from history."""
    return list_tasks(limit=limit, offset=offset)


@router.get("/task/{task_id}", tags=["Tasks"])
def get_task_detail(task_id: int, _auth: None = Depends(_require_api_key)) -> dict[str, Any]:
    """Return full details for a single task including all step results."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task["step_results"] = get_step_results(task_id)
    return task
