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
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

from core.llm import is_ollama_available, list_available_models
from core.memory import (
    append_async_job_event,
    cleanup_async_jobs,
    create_async_job,
    get_async_job,
    get_step_results,
    get_task,
    init_db,
    list_tasks,
    prune_knowledge,
    request_async_job_cancellation,
    reset_orphaned_async_jobs,
    set_async_job_state,
    start_async_job,
)
from core.orchestrator import Orchestrator, OrchestratorCancelledError
from core.settings import SETTINGS

logger = logging.getLogger(__name__)
router = APIRouter()
_EXECUTOR = ThreadPoolExecutor(max_workers=SETTINGS.max_workers, thread_name_prefix="synapse-job")
_MAX_RETENTION_HOURS = 24 * 365


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RunTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=5, max_length=2000, description="The user's high-level goal")
    model: str = Field("mistral", description="Ollama model name to use")
    enable_reflection: bool = Field(True, description="Run reflector agent after each step")


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


class CancelJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class CleanupJobsResponse(BaseModel):
    removed: int
    retention_hours: int


class PruneKBResponse(BaseModel):
    removed: int
    max_entries: int


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    model: str


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: str


class EvalResponse(BaseModel):
    goal: str
    model: str
    status: str
    elapsed_seconds: float
    step_count: int
    estimated_complexity: str
    final_output_present: bool
    step_results_count: int


# ---------------------------------------------------------------------------
# Lifecycle helpers (called from main.py lifespan)
# ---------------------------------------------------------------------------


def startup_init() -> None:
    """Initialize database and recover orphaned async jobs on startup."""
    init_db()
    reset = reset_orphaned_async_jobs()
    if reset:
        logger.info("Reset %d orphaned async jobs to failed on startup", reset)


def shutdown_executor() -> None:
    """Gracefully shut down the thread pool executor."""
    _EXECUTOR.shutdown(wait=True)
    logger.info("Thread pool executor shut down")


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory token bucket per client IP)
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW_S = 60.0
_RATE_LIMIT_MAX_REQUESTS = 30
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _rate_limit(request: Request) -> None:
    """FastAPI dependency — 429 if client exceeds the request cap."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_S
    bucket = _rate_buckets[client_ip]
    # Prune expired entries
    bucket[:] = [t for t in bucket if t > window_start]
    if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    bucket.append(now)


# ---------------------------------------------------------------------------
# Input hardening
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")


def _sanitize_goal(goal: str) -> str:
    """Strip control characters and leading/trailing whitespace from goal."""
    return _CONTROL_CHAR_RE.sub("", goal).strip()


# ---------------------------------------------------------------------------
# Request metrics (simple in-memory counters for observability)
# ---------------------------------------------------------------------------

_request_metrics: dict[str, Any] = {
    "total_requests": 0,
    "total_errors": 0,
    "route_counts": defaultdict(int),
    "route_errors": defaultdict(int),
}


def _track_request(route: str, status_code: int) -> None:
    _request_metrics["total_requests"] += 1
    _request_metrics["route_counts"][route] += 1
    if status_code >= 400:
        _request_metrics["total_errors"] += 1
        _request_metrics["route_errors"][route] += 1


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = SETTINGS.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _require_api_key_sse(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
) -> None:
    """Auth dependency that also accepts api_key query param (for EventSource)."""
    expected = SETTINGS.api_key
    if not expected:
        return
    provided = x_api_key or api_key
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _record_event(job_id: str, event: str, data: dict[str, Any]) -> None:
    append_async_job_event(job_id, event, data)


def _resolve_retention_hours(retention_hours: int | None) -> int:
    if not isinstance(retention_hours, int):
        return SETTINGS.async_job_retention_hours
    return retention_hours


def _run_retention_cleanup(retention_hours: int | None = None) -> None:
    cleanup_async_jobs(retention_hours=_resolve_retention_hours(retention_hours))


def _run_job(job_id: str, request: RunTaskRequest) -> None:
    def on_progress(event: str, data: dict[str, Any]) -> None:
        _record_event(job_id, event, data)

    def cancellation_check() -> bool:
        job = get_async_job(job_id)
        return bool(job and job.get("status") == "cancelling")

    try:
        start_status = start_async_job(job_id)
        if start_status is None:
            return
        if start_status != "running":
            return

        orchestrator = Orchestrator(
            model=request.model,
            enable_reflection=request.enable_reflection,
            interactive=False,
            progress_callback=on_progress,
            cancellation_check=cancellation_check,
        )
        result = orchestrator.run(request.goal)
        if cancellation_check():
            _record_event(job_id, "cancelled", {"message": "Cancelled by user request."})
            set_async_job_state(job_id, status="cancelled", result=None, error="Cancelled by user request.")
            return
        set_async_job_state(job_id, status="completed", result=result, error=None)
    except OrchestratorCancelledError:
        _record_event(job_id, "cancelled", {"message": "Cancelled by user request."})
        set_async_job_state(job_id, status="cancelled", result=None, error="Cancelled by user request.")
    except Exception as exc:
        logger.exception("Async pipeline error: %s", exc)
        set_async_job_state(job_id, status="failed", result=None, error=str(exc))


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
def run_task(
    request: RunTaskRequest,
    _auth: None = Depends(_require_api_key),
    _rate: None = Depends(_rate_limit),
) -> RunTaskResponse:
    """Execute a user goal through the full Planner→Researcher→Executor→Reflector pipeline.

    This call is **synchronous** – it blocks until the pipeline completes.
    API execution is always non-interactive; use CLI mode for manual plan approval.
    """
    goal = _sanitize_goal(request.goal)
    logger.info("POST /run-task – goal=%s", goal[:80])

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
        result = orchestrator.run(goal)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunTaskResponse(**result)


@router.post("/run-task-async", response_model=AsyncRunTaskResponse, tags=["Tasks"])
def run_task_async(
    request: RunTaskRequest,
    retention_hours: int | None = Query(default=None, ge=1, le=_MAX_RETENTION_HOURS),
    _auth: None = Depends(_require_api_key),
    _rate: None = Depends(_rate_limit),
) -> AsyncRunTaskResponse:
    """Submit a task for background execution and return a job id."""
    goal = _sanitize_goal(request.goal)
    _run_retention_cleanup(retention_hours)

    job_id = str(uuid.uuid4())
    create_async_job(
        job_id=job_id,
        goal=goal,
        model=request.model,
        status="pending",
    )
    _EXECUTOR.submit(_run_job, job_id, request)
    return AsyncRunTaskResponse(job_id=job_id, status="queued", goal=goal)


@router.get("/run-task-async/{job_id}", response_model=JobStatusResponse, tags=["Tasks"])
def run_task_status(
    job_id: str,
    retention_hours: int | None = Query(default=None, ge=1, le=_MAX_RETENTION_HOURS),
    _auth: None = Depends(_require_api_key),
) -> JobStatusResponse:
    """Return background job status, events, and final result when done."""
    _run_retention_cleanup(retention_hours)
    job = get_async_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(**job)


@router.post("/run-task-async/{job_id}/cancel", response_model=CancelJobResponse, tags=["Tasks"])
def cancel_run_task(
    job_id: str,
    retention_hours: int | None = Query(default=None, ge=1, le=_MAX_RETENTION_HOURS),
    _auth: None = Depends(_require_api_key),
) -> CancelJobResponse:
    """Request cancellation for a queued/running background task."""
    _run_retention_cleanup(retention_hours)

    status, transitioned = request_async_job_cancellation(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if status in {"completed", "failed"}:
        raise HTTPException(status_code=409, detail=f"Job {job_id} already {status}")

    if status == "running":
        status, transitioned = request_async_job_cancellation(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if status == "cancelling":
        if transitioned:
            _record_event(job_id, "cancellation_requested", {"message": "Cancellation requested."})
        return CancelJobResponse(job_id=job_id, status="cancelling", message="Cancellation requested.")

    if status == "cancelled":
        if transitioned:
            _record_event(job_id, "cancelled", {"message": "Cancelled by user request."})
        return CancelJobResponse(job_id=job_id, status="cancelled", message="Job cancelled.")

    return CancelJobResponse(job_id=job_id, status=status, message=f"Job status is {status}.")


@router.post("/run-task-async/cleanup", response_model=CleanupJobsResponse, tags=["Tasks"])
def cleanup_run_task_jobs(
    retention_hours: int = Query(default=SETTINGS.async_job_retention_hours, ge=1, le=_MAX_RETENTION_HOURS),
    _auth: None = Depends(_require_api_key),
) -> CleanupJobsResponse:
    """Delete terminal async jobs older than a configurable retention window."""
    removed = cleanup_async_jobs(retention_hours=retention_hours)
    return CleanupJobsResponse(removed=removed, retention_hours=retention_hours)


@router.post("/knowledge/prune", response_model=PruneKBResponse, tags=["Knowledge"])
def prune_knowledge_base(
    max_entries: int = Query(default=SETTINGS.kb_max_entries, ge=0, le=100000),
    _auth: None = Depends(_require_api_key),
) -> PruneKBResponse:
    """Prune the knowledge base to max_entries by removing lowest-quality items."""
    removed = prune_knowledge(max_entries=max_entries)
    return PruneKBResponse(removed=removed, max_entries=max_entries)


def _sse_event(event_data: dict[str, Any]) -> str:
    """Format a dict as an SSE ``data:`` frame."""
    import json as _json

    return f"data: {_json.dumps(event_data)}\n\n"


@router.get("/run-task-async/{job_id}/stream", tags=["Tasks"])
def stream_task_events(
    job_id: str,
    _auth: None = Depends(_require_api_key_sse),
) -> StreamingResponse:
    """Stream async job events via Server-Sent Events (SSE).

    Each event is a JSON object sent as a ``data:`` frame. The stream closes
    when the job reaches a terminal state (completed, failed, or cancelled).
    """
    import json as _json
    import time as _time

    def _event_generator():
        seen = 0
        while True:
            job = get_async_job(job_id)
            if job is None:
                yield _sse_event({"error": f"Job {job_id} not found"})
                return
            events = job.get("events", [])
            while seen < len(events):
                yield _sse_event(events[seen])
                seen += 1
            if job["status"] in ("completed", "failed", "cancelled"):
                yield _sse_event({"status": job["status"], "done": True})
                return
            _time.sleep(1.0)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@router.post("/eval", response_model=EvalResponse, tags=["Eval"])
def run_eval(
    goal: str = Query("Build a simple calculator web app", min_length=5, max_length=2000),
    model: str = Query(SETTINGS.default_model),
    _auth: None = Depends(_require_api_key),
) -> EvalResponse:
    """Run the pipeline with a synthetic goal and return evaluation metrics."""
    from eval_harness import run_evaluation

    metrics = run_evaluation(goal=goal, model=model)
    return EvalResponse(**metrics)


@router.get("/metrics", tags=["Observability"])
def get_metrics(_auth: None = Depends(_require_api_key)) -> dict[str, Any]:
    """Return basic request metrics for observability."""
    return {
        "total_requests": _request_metrics["total_requests"],
        "total_errors": _request_metrics["total_errors"],
        "route_counts": dict(_request_metrics["route_counts"]),
        "route_errors": dict(_request_metrics["route_errors"]),
    }
