"""
core/memory.py – Persistent task history using SQLite.

Provides a simple interface to store and retrieve past tasks, agent outputs,
and per-task step results so agents can reuse previous knowledge.
"""

from __future__ import annotations

import json
import logging
import sqlite3
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
    """Create tables if they do not already exist."""
    with _get_conn() as conn:
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
                created_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status_created_at
                ON tasks(status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                ON tasks(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_step_results_task_id_step_index
                ON step_results(task_id, step_index);
            CREATE INDEX IF NOT EXISTS idx_knowledge_keyword
                ON knowledge_base(keyword);
            """
        )
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


def update_task(
    task_id: int,
    status: str | None = None,
    plan: dict[str, Any] | None = None,
    final_output: dict[str, Any] | None = None,
) -> None:
    """Partial update a task record."""
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_now()]

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if plan is not None:
        fields.append("plan = ?")
        values.append(json.dumps(plan))
    if final_output is not None:
        fields.append("final_output = ?")
        values.append(json.dumps(final_output))

    values.append(task_id)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?",  # noqa: S608
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
# Knowledge base
# ---------------------------------------------------------------------------


def store_knowledge(keyword: str, content: str, source_task: int | None = None) -> None:
    """Store a reusable piece of knowledge."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO knowledge_base (keyword, content, source_task, created_at) VALUES (?,?,?,?)",
            (keyword.lower(), content, source_task, _now()),
        )


def search_knowledge(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Simple keyword search over the knowledge base."""
    terms = query.lower().split()
    if not terms:
        return []
    conditions = " OR ".join(["keyword LIKE ?"] * len(terms))
    params = [f"%{t}%" for t in terms]
    params.append(limit)
    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM knowledge_base WHERE {conditions} ORDER BY id DESC LIMIT ?",  # noqa: S608
            params,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # Deserialize JSON fields
    for field in ("plan", "final_output", "research", "execution", "reflection"):
        if field in d and d[field] is not None:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
