"""
core/memory.py – Persistent task history using SQLite.

Provides a simple interface to store and retrieve past tasks, agent outputs,
and per-task step results so agents can reuse previous knowledge.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "synapse_memory.db"


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------


def _ensure_db_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not already exist.

    Enables WAL journal mode and sets a busy-timeout so concurrent writers
    wait briefly instead of immediately raising ``database is locked``.
    """
    with _get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                goal        TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'pending',
                plan        TEXT,
                final_output TEXT,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS step_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     INTEGER NOT NULL REFERENCES tasks(id),
                step_index  INTEGER NOT NULL,
                step_text   TEXT    NOT NULL,
                research    TEXT,
                execution   TEXT,
                reflection  TEXT,
                status      TEXT    NOT NULL DEFAULT 'pending',
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_base (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword     TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                source_task INTEGER REFERENCES tasks(id),
                quality_score REAL  DEFAULT NULL,
                provenance  TEXT    DEFAULT NULL,
                verified    INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS async_jobs (
                job_id       TEXT PRIMARY KEY,
                goal         TEXT NOT NULL,
                model        TEXT NOT NULL,
                status       TEXT NOT NULL,
                events       TEXT NOT NULL DEFAULT '[]',
                result       TEXT,
                error        TEXT,
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status_created_at
                ON tasks(status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                ON tasks(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_step_results_task_id_step_index
                ON step_results(task_id, step_index);
            CREATE INDEX IF NOT EXISTS idx_knowledge_keyword
                ON knowledge_base(keyword);
            CREATE INDEX IF NOT EXISTS idx_async_jobs_status_updated_at
                ON async_jobs(status, updated_at DESC);
            """
        )
        # Migrations for existing databases (safe to run repeatedly)
        for col, typedef in [
            ("quality_score", "REAL DEFAULT NULL"),
            ("provenance", "TEXT DEFAULT NULL"),
            ("verified", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE knowledge_base ADD COLUMN {col} {typedef}")  # noqa: S608
            except sqlite3.OperationalError:
                pass  # Column already exists
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------


def create_task(goal: str) -> int:
    """Insert a new task and return its id."""
    now = _now()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (goal, status, created_at, updated_at) VALUES (?,?,?,?)",
            (goal, "pending", now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


_TASK_UPDATE_COLUMNS: set[str] = {"status", "plan", "final_output", "updated_at"}


def update_task(
    task_id: int,
    status: str | None = None,
    plan: dict[str, Any] | None = None,
    final_output: dict[str, Any] | None = None,
) -> None:
    """Partial update a task record."""
    assignments: list[str] = []
    values: list[Any] = []

    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if plan is not None:
        assignments.append("plan = ?")
        values.append(json.dumps(plan))
    if final_output is not None:
        assignments.append("final_output = ?")
        values.append(json.dumps(final_output))

    if not assignments:
        return

    assignments.append("updated_at = ?")
    values.append(_now())
    values.append(task_id)

    set_clause = ", ".join(assignments)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE tasks SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )


def get_task(task_id: int) -> dict[str, Any] | None:
    """Return a task record as a dict, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_tasks(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Return most recent tasks using limit/offset pagination."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Step results
# ---------------------------------------------------------------------------


def save_step_result(
    task_id: int,
    step_index: int,
    step_text: str,
    research: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    reflection: dict[str, Any] | None = None,
    status: str = "completed",
) -> int:
    """Persist a step result and return its id."""
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO step_results
               (task_id, step_index, step_text, research, execution, reflection, status, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                task_id,
                step_index,
                step_text,
                json.dumps(research) if research else None,
                json.dumps(execution) if execution else None,
                json.dumps(reflection) if reflection else None,
                status,
                _now(),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_step_results(task_id: int) -> list[dict[str, Any]]:
    """Return all step results for a task ordered by step_index."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM step_results WHERE task_id = ? ORDER BY step_index",
            (task_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Async jobs
# ---------------------------------------------------------------------------


def create_async_job(
    job_id: str,
    goal: str,
    model: str,
    status: str = "pending",
) -> None:
    """Insert a new async job record."""
    now_ts = _now_ts()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO async_jobs
               (job_id, goal, model, status, events, result, error, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (job_id, goal, model, status, "[]", None, None, now_ts, now_ts),
        )


def start_async_job(job_id: str) -> str | None:
    """Move a pending async job to running and return the resulting status."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM async_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None

        current_status = str(row["status"])
        if current_status != "pending":
            return current_status

        cur = conn.execute(
            """UPDATE async_jobs
               SET status = 'running', updated_at = ?
               WHERE job_id = ? AND status = 'pending'""",
            (_now_ts(), job_id),
        )
        if cur.rowcount > 0:
            return "running"

        refreshed = conn.execute(
            "SELECT status FROM async_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return str(refreshed["status"]) if refreshed is not None else None


def get_async_job(job_id: str) -> dict[str, Any] | None:
    """Return an async job record as a dict, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM async_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def set_async_job_state(
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    """Set async job status and terminal payloads. Returns False if job does not exist."""
    with _get_conn() as conn:
        cur = conn.execute(
            """UPDATE async_jobs
               SET status = ?, result = ?, error = ?, updated_at = ?
               WHERE job_id = ?""",
            (
                status,
                json.dumps(result) if result is not None else None,
                error,
                _now_ts(),
                job_id,
            ),
        )
        return cur.rowcount > 0


def request_async_job_cancellation(job_id: str) -> tuple[str | None, bool]:
    """Request cancellation and return (status, transitioned)."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM async_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None, False

        current_status = str(row["status"])
        if current_status == "pending":
            cur = conn.execute(
                """UPDATE async_jobs
                   SET status = 'cancelled', error = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'pending'""",
                ("Cancelled by user request.", _now_ts(), job_id),
            )
            if cur.rowcount > 0:
                return "cancelled", True
            refreshed = conn.execute(
                "SELECT status FROM async_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return (str(refreshed["status"]) if refreshed is not None else None), False

        if current_status == "running":
            cur = conn.execute(
                """UPDATE async_jobs
                   SET status = 'cancelling', updated_at = ?
                   WHERE job_id = ? AND status = 'running'""",
                (_now_ts(), job_id),
            )
            if cur.rowcount > 0:
                return "cancelling", True
            refreshed = conn.execute(
                "SELECT status FROM async_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return (str(refreshed["status"]) if refreshed is not None else None), False

        return current_status, False


def append_async_job_event(job_id: str, event: str, data: dict[str, Any]) -> bool:
    """Append one progress event to an async job. Returns False if job does not exist."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT events FROM async_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return False

        raw_events = row["events"]
        try:
            events = json.loads(raw_events) if raw_events else []
        except (json.JSONDecodeError, TypeError):
            events = []
        if not isinstance(events, list):
            events = []

        events.append({"event": event, "ts": _now_ts(), **data})
        conn.execute(
            "UPDATE async_jobs SET events = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(events), _now_ts(), job_id),
        )
        return True


def cleanup_async_jobs(retention_hours: int = 24, now_ts: float | None = None) -> int:
    """Delete terminal async jobs older than retention window and return removed count."""
    current_ts = now_ts if now_ts is not None else _now_ts()
    cutoff = current_ts - (retention_hours * 3600)
    with _get_conn() as conn:
        cur = conn.execute(
            """DELETE FROM async_jobs
               WHERE status IN ('completed', 'failed', 'cancelled')
               AND updated_at < ?""",
            (cutoff,),
        )
        return cur.rowcount


def reset_orphaned_async_jobs() -> int:
    """Reset any jobs stuck in 'running' or 'cancelling' to 'failed' on startup.

    Returns the number of jobs reset.
    """
    with _get_conn() as conn:
        cur = conn.execute(
            """UPDATE async_jobs
               SET status = 'failed',
                   error = 'Server restarted while job was running.',
                   updated_at = ?
               WHERE status IN ('running', 'cancelling')""",
            (_now_ts(),),
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


def store_knowledge(
    keyword: str,
    content: str,
    source_task: int | None = None,
    quality_score: float | None = None,
    provenance: str | None = None,
) -> None:
    """Store a reusable piece of knowledge.

    Only knowledge with a quality_score above the threshold should be stored
    by callers — this function does not enforce the gate itself.
    """
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO knowledge_base
               (keyword, content, source_task, quality_score, provenance, created_at)
               VALUES (?,?,?,?,?,?)""",
            (keyword.lower(), content, source_task, quality_score, provenance, _now()),
        )


def search_knowledge(query: str, limit: int = 5, min_quality: float = 0.0) -> list[dict[str, Any]]:
    """Search the knowledge base, filtering by minimum quality score.

    Results are ordered by quality_score DESC (NULLs last), then by id DESC.
    """
    terms = query.lower().split()
    if not terms:
        return []
    # Build parameterised conditions from a fixed column name — no user input
    # reaches the SQL template itself.
    clause = " OR ".join(["keyword LIKE ?"] * len(terms))
    params: list[Any] = [f"%{t}%" for t in terms]
    params.append(min_quality)
    params.append(limit)
    sql = (
        "SELECT * FROM knowledge_base WHERE ("
        + clause
        + ") AND (quality_score IS NULL OR quality_score >= ?) "
        "ORDER BY quality_score DESC NULLS LAST, id DESC LIMIT ?"
    )
    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def prune_knowledge(max_entries: int) -> int:
    """Remove lowest-quality entries exceeding max_entries. Returns count removed."""
    if max_entries <= 0:
        return 0
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]  # type: ignore[union-attr]
        if count <= max_entries:
            return 0
        to_delete = count - max_entries
        conn.execute(
            """DELETE FROM knowledge_base WHERE id IN (
                SELECT id FROM knowledge_base
                ORDER BY quality_score ASC NULLS FIRST, id ASC
                LIMIT ?
            )""",
            (to_delete,),
        )
    return to_delete


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return time.time()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # Deserialize JSON fields
    for field in (
        "plan",
        "final_output",
        "research",
        "execution",
        "reflection",
        "events",
        "result",
    ):
        if field in d and d[field] is not None:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
