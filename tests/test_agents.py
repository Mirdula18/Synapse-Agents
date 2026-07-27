"""
tests/test_agents.py – Unit tests for all agents and core modules.

Tests mock the Ollama API so the suite runs fully offline without a
running Ollama instance.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response returning *payload* as JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": json.dumps(payload)}
    return mock_resp


# ---------------------------------------------------------------------------
# core.llm
# ---------------------------------------------------------------------------


class TestGenerateResponse:
    def test_returns_parsed_dict(self):
        from core.llm import generate_response

        payload = {"goal": "test", "steps": ["step1"], "estimated_complexity": "low"}
        with patch("requests.post", return_value=_make_ollama_response(payload)):
            result = generate_response("hello", system_role="planner")
        assert result["goal"] == "test"
        assert "steps" in result

    def test_confidence_injected(self):
        from core.llm import generate_response

        payload = {"result": "x" * 300}
        with patch("requests.post", return_value=_make_ollama_response(payload)):
            result = generate_response("p", system_role="executor")
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_retries_on_failure_then_succeeds(self):
        from core.llm import generate_response
        import requests as req

        good = _make_ollama_response({"result": "ok"})
        with patch("requests.post", side_effect=[req.RequestException("boom"), good]):
            result = generate_response("p", system_role="executor", retries=3, backoff=0)
        assert result["result"] == "ok"

    def test_raises_after_all_retries(self):
        from core.llm import generate_response
        import requests as req

        with patch("requests.post", side_effect=req.RequestException("always fails")):
            with pytest.raises(RuntimeError, match="All"):
                generate_response("p", retries=2, backoff=0)

    def test_strips_markdown_fences(self):
        from core.llm import _extract_json

        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_json_from_prose(self):
        from core.llm import _extract_json

        text = 'Here is the result: {"answer": 42} hope that helps'
        result = _extract_json(text)
        assert result == {"answer": 42}

    def test_raises_when_no_json(self):
        from core.llm import _extract_json

        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("no json here at all")

    def test_handles_ndjson_payload_when_response_json_fails(self):
        from core.llm import generate_response

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "{", 1)
        mock_resp.text = "\n".join(
            [
                '{"response":"{\\"goal\\":\\"Plan\\","}',
                '{"response":"\\"steps\\":[\\"s1\\"],\\"estimated_complexity\\":\\"low\\"}"}',
            ]
        )

        with patch("requests.post", return_value=mock_resp):
            result = generate_response("hello", system_role="planner")

        assert result["goal"] == "Plan"
        assert result["steps"] == ["s1"]
        assert result["estimated_complexity"] == "low"


# ---------------------------------------------------------------------------
# core.settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_dev_defaults_use_explicit_local_origins_with_credentials(self, monkeypatch):
        from core.settings import load_settings

        monkeypatch.delenv("SYNAPSE_ENV", raising=False)
        monkeypatch.delenv("SYNAPSE_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("SYNAPSE_CORS_ALLOW_CREDENTIALS", raising=False)

        settings = load_settings()
        assert settings.environment == "development"
        assert settings.cors_allow_credentials is True
        assert "*" not in settings.cors_origins
        assert any("localhost" in origin for origin in settings.cors_origins)

    def test_rejects_wildcard_when_credentials_enabled(self, monkeypatch):
        from core.settings import load_settings

        monkeypatch.setenv("SYNAPSE_CORS_ORIGINS", "*")
        monkeypatch.setenv("SYNAPSE_CORS_ALLOW_CREDENTIALS", "true")

        with pytest.raises(ValueError, match="Wildcard"):
            load_settings()

    def test_requires_explicit_origins_in_production_with_credentials(self, monkeypatch):
        from core.settings import load_settings

        monkeypatch.setenv("SYNAPSE_ENV", "production")
        monkeypatch.delenv("SYNAPSE_CORS_ORIGINS", raising=False)
        monkeypatch.setenv("SYNAPSE_CORS_ALLOW_CREDENTIALS", "true")

        with pytest.raises(ValueError, match="Explicit SYNAPSE_CORS_ORIGINS"):
            load_settings()

    def test_allows_wildcard_when_credentials_disabled(self, monkeypatch):
        from core.settings import load_settings

        monkeypatch.setenv("SYNAPSE_CORS_ORIGINS", "*")
        monkeypatch.setenv("SYNAPSE_CORS_ALLOW_CREDENTIALS", "false")

        settings = load_settings()
        assert settings.cors_allow_credentials is False
        assert settings.cors_origins == ["*"]

    def test_invalid_float_env_values_fallback_to_defaults_with_warning(self, monkeypatch, caplog):
        from core.settings import load_settings

        monkeypatch.setenv("OLLAMA_RETRY_BACKOFF_S", "not-a-number")
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "totally-bad")

        caplog.set_level("WARNING")
        settings = load_settings()

        assert settings.ollama_retry_backoff_s == 1.5
        assert settings.ollama_temperature == 0.2
        assert "OLLAMA_RETRY_BACKOFF_S" in caplog.text
        assert "OLLAMA_TEMPERATURE" in caplog.text

    def test_float_env_values_are_clamped_to_bounds_with_warning(self, monkeypatch, caplog):
        from core.settings import load_settings

        monkeypatch.setenv("OLLAMA_RETRY_BACKOFF_S", "-10")
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "99")

        caplog.set_level("WARNING")
        settings = load_settings()

        assert settings.ollama_retry_backoff_s == 0.1
        assert settings.ollama_temperature == 2.0
        assert "clamping to minimum" in caplog.text
        assert "clamping to maximum" in caplog.text


# ---------------------------------------------------------------------------
# agents.planner
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    def _mock_plan(self):
        return {
            "goal": "Build a portfolio site",
            "steps": ["Set up project", "Create HTML", "Add styles"],
            "estimated_complexity": "medium",
        }

    def test_run_returns_valid_plan(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        with patch("agents.planner.generate_response", return_value=self._mock_plan()):
            plan = agent.run("Build a portfolio website")
        assert plan["goal"]
        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) == 3
        assert plan["estimated_complexity"] == "medium"

    def test_validate_falls_back_on_empty_steps(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        result = agent._validate({}, "my goal")
        assert len(result["steps"]) == 1
        assert "my goal" in result["steps"][0]

    def test_validate_normalises_complexity(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        result = agent._validate({"steps": ["s1"], "estimated_complexity": "EXTREME"}, "g")
        assert result["estimated_complexity"] == "medium"

    def test_validate_keeps_valid_complexity(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        for c in ("low", "medium", "high"):
            result = agent._validate({"steps": ["s1"], "estimated_complexity": c}, "g")
            assert result["estimated_complexity"] == c

    def test_timeout_uses_fallback_plan(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        with patch("agents.planner.generate_response", side_effect=RuntimeError("Read timed out")):
            plan = agent.run("Build a portfolio website")
        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) >= 3
        assert plan["estimated_complexity"] in {"low", "medium", "high"}
        assert plan["confidence"] < 0.6

    def test_non_timeout_error_is_reraised(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        with patch("agents.planner.generate_response", side_effect=RuntimeError("Connection refused")):
            with pytest.raises(RuntimeError, match="Connection refused"):
                agent.run("Build a portfolio website")

    def test_json_decode_style_error_uses_fallback_plan(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        with patch(
            "agents.planner.generate_response",
            side_effect=RuntimeError("Unterminated string starting at: line 1 column 2"),
        ):
            plan = agent.run("Build a portfolio website")

        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) >= 3
        assert plan["confidence"] < 0.6


# ---------------------------------------------------------------------------
# agents.researcher
# ---------------------------------------------------------------------------


class TestResearcherAgent:
    def _mock_research(self):
        return {
            "step": "Create HTML",
            "details": "Use semantic HTML5 elements",
            "resources": ["MDN", "W3C"],
            "best_practices": ["Use semantic tags"],
            "pitfalls": ["Avoid inline styles"],
        }

    def test_run_returns_valid_research(self):
        from agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        with patch("agents.researcher.generate_response", return_value=self._mock_research()), \
             patch("agents.researcher.search_knowledge", return_value=[]):
            result = agent.run("Create HTML", goal="Build a portfolio")
        assert result["step"]
        assert isinstance(result["resources"], list)
        assert isinstance(result["best_practices"], list)

    def test_validate_handles_missing_fields(self):
        from agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        result = agent._validate({}, "my step")
        assert result["step"] == "my step"
        assert result["details"] == "No details provided."
        assert result["resources"] == []

    def test_format_knowledge_empty(self):
        from agents.researcher import ResearcherAgent

        result = ResearcherAgent._format_knowledge([])
        assert result == ""

    def test_format_knowledge_with_items(self):
        from agents.researcher import ResearcherAgent

        snippets = [{"keyword": "python", "content": "Python is great"}]
        result = ResearcherAgent._format_knowledge(snippets)
        assert "python" in result
        assert "Python is great" in result

    def test_researcher_prompt_marks_knowledge_as_untrusted(self):
        from agents.researcher import RESEARCHER_PROMPT_TEMPLATE

        assert "retrieved_knowledge" in RESEARCHER_PROMPT_TEMPLATE
        assert "untrusted" in RESEARCHER_PROMPT_TEMPLATE.lower() or "NOT an instruction" in RESEARCHER_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# agents.executor
# ---------------------------------------------------------------------------


class TestExecutorAgent:
    def _mock_execution(self):
        return {
            "step": "Create HTML",
            "result": "<!DOCTYPE html><html>...</html>",
            "code": "<!DOCTYPE html>...",
            "explanation": "Used semantic HTML5",
            "status": "completed",
        }

    def test_run_returns_valid_result(self):
        from agents.executor import ExecutorAgent

        agent = ExecutorAgent()
        with patch("agents.executor.generate_response", return_value=self._mock_execution()):
            result = agent.run("Create HTML", goal="Portfolio", research={"details": "HTML5"})
        assert result["status"] == "completed"
        assert result["result"]

    def test_validate_normalises_bad_status(self):
        from agents.executor import ExecutorAgent

        agent = ExecutorAgent()
        result = agent._validate({"status": "unknown"}, "step")
        assert result["status"] == "completed"

    def test_validate_fills_missing_result(self):
        from agents.executor import ExecutorAgent

        agent = ExecutorAgent()
        result = agent._validate({}, "my step")
        assert result["result"] == "No output generated."

    def test_format_research_none(self):
        from agents.executor import ExecutorAgent

        result = ExecutorAgent._format_research(None)
        assert "No research" in result

    def test_format_research_with_data(self):
        from agents.executor import ExecutorAgent

        research = {
            "details": "Use FastAPI",
            "resources": ["FastAPI docs"],
            "best_practices": ["Use Pydantic"],
            "pitfalls": ["Don't block the event loop"],
        }
        result = ExecutorAgent._format_research(research)
        assert "FastAPI" in result
        assert "Pydantic" in result


# ---------------------------------------------------------------------------
# agents.reflector
# ---------------------------------------------------------------------------


class TestReflectorAgent:
    def _mock_reflection_accepted(self):
        return {
            "step": "Create HTML",
            "original_result": "Good HTML",
            "issues_found": [],
            "improved_result": None,
            "quality_score": 0.9,
            "action_taken": "accepted",
        }

    def _mock_reflection_improved(self):
        return {
            "step": "Create HTML",
            "original_result": "Bad HTML",
            "issues_found": ["Missing doctype"],
            "improved_result": "<!DOCTYPE html>...",
            "quality_score": 0.5,
            "action_taken": "improved",
        }

    def test_run_accepted(self):
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent()
        execution = {"result": "Good HTML"}
        with patch("agents.reflector.generate_response", return_value=self._mock_reflection_accepted()):
            result = agent.run("Create HTML", execution)
        assert result["action_taken"] == "accepted"
        assert result["quality_score"] == 0.9

    def test_run_improved(self):
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent()
        execution = {"result": "Bad HTML"}
        with patch("agents.reflector.generate_response", return_value=self._mock_reflection_improved()):
            result = agent.run("Create HTML", execution)
        assert result["improved_result"] is not None

    def test_validate_clamps_quality_score(self):
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent()
        result = agent._validate({"quality_score": 1.5}, "s", "orig")
        assert result["quality_score"] == 1.0

        result = agent._validate({"quality_score": -0.3}, "s", "orig")
        assert result["quality_score"] == 0.0

    def test_validate_high_quality_forces_accepted(self):
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent(quality_threshold=0.75)
        result = agent._validate(
            {"quality_score": 0.9, "issues_found": [], "action_taken": "improved"},
            "s",
            "orig",
        )
        assert result["action_taken"] == "accepted"
        assert result["improved_result"] is None


# ---------------------------------------------------------------------------
# core.memory
# ---------------------------------------------------------------------------


class TestMemory:
    def setup_method(self):
        """Point DB at a temp file for each test."""
        import tempfile
        from pathlib import Path
        import core.memory as mem

        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        mem.DB_PATH = Path(self._tmp.name)
        mem.init_db()

    def teardown_method(self):
        import os
        import core.memory as mem
        from pathlib import Path

        self._tmp.close()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass
        # Reset to default
        mem.DB_PATH = Path(__file__).parent.parent / "data" / "synapse_memory.db"

    def test_create_and_get_task(self):
        from core.memory import create_task, get_task

        task_id = create_task("Test goal")
        assert isinstance(task_id, int)
        task = get_task(task_id)
        assert task is not None
        assert task["goal"] == "Test goal"
        assert task["status"] == "pending"

    def test_update_task_status(self):
        from core.memory import create_task, get_task, update_task

        task_id = create_task("Goal")
        update_task(task_id, status="completed")
        task = get_task(task_id)
        assert task["status"] == "completed"

    def test_update_task_plan(self):
        from core.memory import create_task, get_task, update_task

        task_id = create_task("Goal")
        plan = {"steps": ["s1", "s2"], "goal": "Goal", "estimated_complexity": "low"}
        update_task(task_id, plan=plan)
        task = get_task(task_id)
        assert task["plan"]["steps"] == ["s1", "s2"]

    def test_list_tasks(self):
        from core.memory import create_task, list_tasks

        create_task("Goal A")
        create_task("Goal B")
        tasks = list_tasks()
        assert len(tasks) >= 2

    def test_save_and_get_step_results(self):
        from core.memory import create_task, get_step_results, save_step_result

        task_id = create_task("Goal")
        save_step_result(
            task_id=task_id,
            step_index=0,
            step_text="Step one",
            research={"details": "research"},
            execution={"result": "done"},
        )
        results = get_step_results(task_id)
        assert len(results) == 1
        assert results[0]["step_text"] == "Step one"
        assert results[0]["research"]["details"] == "research"

    def test_get_nonexistent_task(self):
        from core.memory import get_task

        assert get_task(99999) is None

    def test_store_and_search_knowledge(self):
        from core.memory import search_knowledge, store_knowledge

        store_knowledge("python testing", "Use pytest for Python unit tests")
        results = search_knowledge("python")
        assert len(results) >= 1
        assert any("pytest" in r["content"] for r in results)

    def test_search_knowledge_empty_query(self):
        from core.memory import search_knowledge

        results = search_knowledge("")
        assert results == []

    def test_store_and_search_knowledge_quality_gating(self):
        from core.memory import search_knowledge, store_knowledge

        store_knowledge(
            "quality test",
            "High quality content",
            quality_score=0.9,
            provenance="task=1:step=0",
        )
        store_knowledge(
            "quality test low",
            "Low quality content",
            quality_score=0.2,
        )
        # Search with min_quality filters out low quality
        results = search_knowledge("quality", min_quality=0.5)
        assert len(results) == 1
        assert results[0]["quality_score"] == 0.9
        assert results[0]["provenance"] == "task=1:step=0"

    def test_store_knowledge_preserves_provenance(self):
        from core.memory import search_knowledge, store_knowledge

        store_knowledge(
            "provenance test",
            "Content with provenance",
            quality_score=1.0,
            provenance="task=42:step=1",
        )
        results = search_knowledge("provenance")
        assert len(results) == 1
        assert results[0]["provenance"] == "task=42:step=1"

    def test_prune_knowledge_removes_lowest_quality(self):
        from core.memory import prune_knowledge, search_knowledge, store_knowledge

        for i in range(5):
            store_knowledge(
                f"prune test {i}",
                f"Content {i}",
                quality_score=float(i) / 4.0,
            )
        removed = prune_knowledge(max_entries=3)
        assert removed == 2
        remaining = search_knowledge("prune", min_quality=0.0)
        assert len(remaining) == 3

    def test_prune_knowledge_noop_when_under_limit(self):
        from core.memory import prune_knowledge, store_knowledge

        store_knowledge("prune noop", "Content", quality_score=0.5)
        removed = prune_knowledge(max_entries=100)
        assert removed == 0

    def test_async_job_lifecycle(self):
        from core.memory import (
            append_async_job_event,
            create_async_job,
            get_async_job,
            set_async_job_state,
        )

        create_async_job(job_id="job-1", goal="Build async", model="mistral", status="pending")

        assert append_async_job_event("job-1", "planning_start", {"goal": "Build async"})
        assert set_async_job_state("job-1", status="running")
        assert set_async_job_state(
            "job-1",
            status="completed",
            result={"task_id": 10, "status": "completed"},
            error=None,
        )

        job = get_async_job("job-1")
        assert job is not None
        assert job["status"] == "completed"
        assert job["result"]["task_id"] == 10
        assert isinstance(job["events"], list)
        assert job["events"][0]["event"] == "planning_start"
        assert isinstance(job["created_at"], float)
        assert isinstance(job["updated_at"], float)

    def test_cleanup_async_jobs_removes_only_expired_terminal_jobs(self):
        from core.memory import cleanup_async_jobs, create_async_job, get_async_job, set_async_job_state

        create_async_job(job_id="done-job", goal="Done", model="mistral", status="pending")
        create_async_job(job_id="running-job", goal="Running", model="mistral", status="pending")
        set_async_job_state("done-job", status="completed", result={"ok": True}, error=None)
        set_async_job_state("running-job", status="running")

        removed = cleanup_async_jobs(retention_hours=24, now_ts=10_000_000_000)
        assert removed == 1
        assert get_async_job("done-job") is None
        assert get_async_job("running-job") is not None

    def test_request_async_job_cancellation_transitions_pending_and_running(self):
        from core.memory import (
            create_async_job,
            get_async_job,
            request_async_job_cancellation,
            set_async_job_state,
        )

        create_async_job(job_id="pending-job", goal="Pending", model="mistral", status="pending")
        status, changed = request_async_job_cancellation("pending-job")
        assert status == "cancelled"
        assert changed is True
        pending_job = get_async_job("pending-job")
        assert pending_job is not None
        assert pending_job["status"] == "cancelled"

        create_async_job(job_id="running-job", goal="Running", model="mistral", status="pending")
        set_async_job_state("running-job", status="running")
        status, changed = request_async_job_cancellation("running-job")
        assert status == "cancelling"
        assert changed is True
        running_job = get_async_job("running-job")
        assert running_job is not None
        assert running_job["status"] == "cancelling"

    def test_start_async_job_does_not_restart_cancelled_job(self):
        from core.memory import create_async_job, request_async_job_cancellation, start_async_job

        create_async_job(job_id="cancelled-job", goal="Cancelled", model="mistral", status="pending")
        request_async_job_cancellation("cancelled-job")
        state = start_async_job("cancelled-job")
        assert state == "cancelled"

    def test_reset_orphaned_async_jobs(self):
        from core.memory import (
            create_async_job,
            get_async_job,
            reset_orphaned_async_jobs,
            set_async_job_state,
        )

        create_async_job(job_id="orphan-run", goal="Orphan", model="mistral", status="pending")
        set_async_job_state("orphan-run", status="running")

        create_async_job(job_id="orphan-cancel", goal="Orphan", model="mistral", status="pending")
        set_async_job_state("orphan-cancel", status="running")
        from core.memory import request_async_job_cancellation
        request_async_job_cancellation("orphan-cancel")

        create_async_job(job_id="done-job", goal="Done", model="mistral", status="pending")
        set_async_job_state("done-job", status="completed", result={"ok": True}, error=None)

        reset = reset_orphaned_async_jobs()
        assert reset == 2

        orphan_run = get_async_job("orphan-run")
        assert orphan_run["status"] == "failed"
        assert "Server restarted" in orphan_run["error"]

        orphan_cancel = get_async_job("orphan-cancel")
        assert orphan_cancel["status"] == "failed"

        done = get_async_job("done-job")
        assert done["status"] == "completed"


# ---------------------------------------------------------------------------
# core.orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_is_timeout_error_detects_requests_timeout(self):
        import requests as req
        from core.orchestrator import _is_timeout_error

        assert _is_timeout_error(req.Timeout("Read timed out")) is True

    def test_is_timeout_error_detects_timeout_error_type(self):
        from core.orchestrator import _is_timeout_error

        assert _is_timeout_error(TimeoutError("boom")) is True

    def test_is_timeout_error_falls_back_to_string_matching(self):
        from core.orchestrator import _is_timeout_error

        assert _is_timeout_error(RuntimeError("Read timed out")) is True
        assert _is_timeout_error(RuntimeError("connection timeout")) is True

    def test_is_timeout_error_returns_false_for_non_timeout(self):
        from core.orchestrator import _is_timeout_error

        assert _is_timeout_error(RuntimeError("connection refused")) is False
        assert _is_timeout_error(ValueError("bad input")) is False

    def test_marks_step_failed_when_executor_returns_failed_status(self, tmp_path):
        import core.memory as mem
        from core.memory import get_step_results
        from core.orchestrator import Orchestrator

        original_db_path = mem.DB_PATH
        mem.DB_PATH = tmp_path / "orchestrator_test.db"

        try:
            orch = Orchestrator(model="mistral", enable_reflection=True)
            orch.planner = MagicMock()
            orch.researcher = MagicMock()
            orch.executor = MagicMock()
            orch.reflector = MagicMock()

            orch.planner.run.return_value = {
                "goal": "Build feature",
                "steps": ["Implement feature"],
                "estimated_complexity": "low",
            }
            orch.researcher.run.return_value = {
                "step": "Implement feature",
                "details": "Use robust validation.",
                "resources": [],
                "best_practices": [],
                "pitfalls": [],
            }
            orch.executor.run.return_value = {
                "step": "Implement feature",
                "result": "Execution failed: invalid schema",
                "status": "failed",
                "confidence": 0.2,
            }

            result = orch.run("Build feature")
            step_results = get_step_results(result["task_id"])
        finally:
            mem.DB_PATH = original_db_path

        assert result["final_output"]["completed_steps"] == 0
        assert result["final_output"]["failed_steps"] == 1
        assert result["final_output"]["overall_status"] == "partial"

        step_output = result["final_output"]["step_outputs"][0]
        assert step_output["status"] == "failed"
        assert "Execution failed" in step_output["result"]

        assert len(step_results) == 1
        assert step_results[0]["status"] == "failed"

        orch.reflector.run.assert_not_called()

    def test_marks_task_cancelled_when_cancellation_check_requests_stop(self, tmp_path):
        import core.memory as mem
        from core.memory import list_tasks
        from core.orchestrator import Orchestrator, OrchestratorCancelledError

        original_db_path = mem.DB_PATH
        mem.DB_PATH = tmp_path / "orchestrator_cancel_test.db"

        try:
            orch = Orchestrator(model="mistral", cancellation_check=lambda: True)
            orch.planner = MagicMock()
            orch.planner.run.return_value = {
                "goal": "Cancel goal",
                "steps": ["s1"],
                "estimated_complexity": "low",
            }

            with pytest.raises(OrchestratorCancelledError, match="Cancelled"):
                orch.run("Cancel goal")

            tasks = list_tasks(limit=1)
            assert tasks[0]["status"] == "cancelled"
        finally:
            mem.DB_PATH = original_db_path

    def test_marks_step_failed_when_executor_status_is_case_insensitive(self, tmp_path):
        import core.memory as mem
        from core.memory import get_step_results
        from core.orchestrator import Orchestrator

        original_db_path = mem.DB_PATH
        mem.DB_PATH = tmp_path / "orchestrator_test_case_insensitive.db"

        try:
            orch = Orchestrator(model="mistral", enable_reflection=True)
            orch.planner = MagicMock()
            orch.researcher = MagicMock()
            orch.executor = MagicMock()
            orch.reflector = MagicMock()

            orch.planner.run.return_value = {
                "goal": "Build feature",
                "steps": ["Implement feature"],
                "estimated_complexity": "low",
            }
            orch.researcher.run.return_value = {
                "step": "Implement feature",
                "details": "Use robust validation.",
                "resources": [],
                "best_practices": [],
                "pitfalls": [],
            }
            orch.executor.run.return_value = {
                "step": "Implement feature",
                "result": "Execution failed: uppercase status",
                "status": "FAILED",
                "confidence": 0.2,
            }

            result = orch.run("Build feature")
            step_results = get_step_results(result["task_id"])
        finally:
            mem.DB_PATH = original_db_path

        assert result["final_output"]["completed_steps"] == 0
        assert result["final_output"]["failed_steps"] == 1
        assert result["final_output"]["overall_status"] == "partial"

        step_output = result["final_output"]["step_outputs"][0]
        assert step_output["status"] == "failed"
        assert "Execution failed" in step_output["result"]

        assert len(step_results) == 1
        assert step_results[0]["status"] == "failed"

        orch.reflector.run.assert_not_called()

    def test_marks_step_failed_when_executor_returns_non_dict_payload(self, tmp_path):
        import core.memory as mem
        from core.memory import get_step_results
        from core.orchestrator import Orchestrator

        original_db_path = mem.DB_PATH
        mem.DB_PATH = tmp_path / "orchestrator_test_invalid_payload.db"

        try:
            orch = Orchestrator(model="mistral", enable_reflection=True)
            orch.planner = MagicMock()
            orch.researcher = MagicMock()
            orch.executor = MagicMock()
            orch.reflector = MagicMock()

            orch.planner.run.return_value = {
                "goal": "Build feature",
                "steps": ["Implement feature"],
                "estimated_complexity": "low",
            }
            orch.researcher.run.return_value = {
                "step": "Implement feature",
                "details": "Use robust validation.",
                "resources": [],
                "best_practices": [],
                "pitfalls": [],
            }
            orch.executor.run.return_value = "invalid payload"

            result = orch.run("Build feature")
            step_results = get_step_results(result["task_id"])
        finally:
            mem.DB_PATH = original_db_path

        assert result["final_output"]["completed_steps"] == 0
        assert result["final_output"]["failed_steps"] == 1
        assert result["final_output"]["overall_status"] == "partial"

        step_output = result["final_output"]["step_outputs"][0]
        assert step_output["status"] == "failed"
        assert "invalid payload type" in step_output["result"].lower()

        assert len(step_results) == 1
        assert step_results[0]["status"] == "failed"

        orch.reflector.run.assert_not_called()

    def test_keeps_step_completed_when_executor_returns_completed_status(self, tmp_path):
        import core.memory as mem
        from core.memory import get_step_results
        from core.orchestrator import Orchestrator

        original_db_path = mem.DB_PATH
        mem.DB_PATH = tmp_path / "orchestrator_test_completed_status.db"

        try:
            orch = Orchestrator(model="mistral", enable_reflection=False)
            orch.planner = MagicMock()
            orch.researcher = MagicMock()
            orch.executor = MagicMock()

            orch.planner.run.return_value = {
                "goal": "Build feature",
                "steps": ["Implement feature"],
                "estimated_complexity": "low",
            }
            orch.researcher.run.return_value = {
                "step": "Implement feature",
                "details": "Use robust validation.",
                "resources": [],
                "best_practices": [],
                "pitfalls": [],
            }
            orch.executor.run.return_value = {
                "step": "Implement feature",
                "result": "Feature implemented successfully",
                "status": "completed",
                "confidence": 0.9,
            }

            result = orch.run("Build feature")
            step_results = get_step_results(result["task_id"])
        finally:
            mem.DB_PATH = original_db_path

        assert result["final_output"]["completed_steps"] == 1
        assert result["final_output"]["failed_steps"] == 0
        assert result["final_output"]["overall_status"] == "completed"

        step_output = result["final_output"]["step_outputs"][0]
        assert step_output["status"] == "completed"
        assert "implemented successfully" in step_output["result"].lower()

        assert len(step_results) == 1
        assert step_results[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# utils.helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_truncate_short_string(self):
        from utils.helpers import truncate

        assert truncate("hello", 10) == "hello"

    def test_truncate_long_string(self):
        from utils.helpers import truncate

        result = truncate("a" * 600, 500)
        assert len(result) < 600
        assert "truncated" in result

    def test_safe_json_dumps_basic(self):
        from utils.helpers import safe_json_dumps

        result = safe_json_dumps({"key": "value"})
        assert '"key"' in result

    def test_safe_json_dumps_non_serialisable(self):
        from utils.helpers import safe_json_dumps

        result = safe_json_dumps(object())
        assert result  # Should not raise

    def test_extract_code_blocks(self):
        from utils.helpers import extract_code_blocks

        text = "Here:\n```python\nprint('hello')\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert "print" in blocks[0]["code"]

    def test_extract_code_blocks_no_blocks(self):
        from utils.helpers import extract_code_blocks

        assert extract_code_blocks("no code here") == []

    def test_safe_shell_disabled_by_default(self):
        from utils.helpers import safe_shell

        result = safe_shell(["echo", "hello"])
        assert result["return_code"] != 0
        assert "disabled" in result["stderr"].lower()

    def test_safe_shell_blocked_command(self):
        from utils.helpers import safe_shell

        with patch("core.settings.SETTINGS") as mock_settings:
            mock_settings.enable_exec = True
            result = safe_shell(["rm", "-rf", "/"])
        assert result["return_code"] != 0
        assert "sandbox" in result["stderr"]

    def test_safe_shell_allowed_command(self):
        from utils.helpers import safe_shell

        with patch("core.settings.SETTINGS") as mock_settings:
            mock_settings.enable_exec = True
            result = safe_shell(["echo", "hello"])
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]

    def test_safe_shell_empty_command(self):
        from utils.helpers import safe_shell

        with patch("core.settings.SETTINGS") as mock_settings:
            mock_settings.enable_exec = True
            result = safe_shell([])
        assert result["return_code"] != 0

    def test_read_file_not_found(self):
        from utils.helpers import read_file

        result = read_file("/nonexistent/path/file.txt")
        assert "[ERROR]" in result

    def test_write_and_read_file(self, tmp_path):
        from utils.helpers import read_file, write_file

        p = tmp_path / "test.txt"
        assert write_file(p, "hello world")
        content = read_file(p)
        assert content == "hello world"


# ---------------------------------------------------------------------------
# api routes (no Ollama needed – mock orchestrator)
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    @pytest.fixture
    def client(self, tmp_path):
        import core.memory as mem

        mem.DB_PATH = tmp_path / "test.db"
        mem.init_db()

        from fastapi.testclient import TestClient
        from main import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        with patch("api.routes.is_ollama_available", return_value=False):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ollama_available"] is False

    def test_models_endpoint(self, client):
        with patch("api.routes.list_available_models", return_value=["mistral", "llama3"]):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "mistral" in data["models"]
        assert "default_model" in data

    def test_history_empty(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_run_task_success(self, client):
        mock_result = {
            "task_id": 1,
            "goal": "Build portfolio",
            "status": "completed",
            "plan": {"goal": "Build portfolio", "steps": ["s1"], "estimated_complexity": "low"},
            "final_output": {"goal": "Build portfolio", "total_steps": 1, "completed_steps": 1, "failed_steps": 0, "step_outputs": [], "overall_status": "completed"},
            "elapsed_seconds": 1.5,
        }
        with patch("api.routes.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = mock_result
            resp = client.post("/run-task", json={"goal": "Build portfolio website"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["goal"] == "Build portfolio"

    def test_run_task_rejects_interactive_field(self, client):
        resp = client.post(
            "/run-task",
            json={
                "goal": "Build portfolio website",
                "interactive": True,
            },
        )

        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body

    def test_run_task_pipeline_error(self, client):
        with patch("api.routes.Orchestrator") as MockOrch:
            MockOrch.return_value.run.side_effect = RuntimeError("LLM offline")
            resp = client.post("/run-task", json={"goal": "Do something interesting"})
        assert resp.status_code == 500

    def test_run_task_async_submit_and_poll(self, client):
        mock_result = {
            "task_id": 99,
            "goal": "Async build",
            "status": "completed",
            "plan": {"goal": "Async build", "steps": ["s1"], "estimated_complexity": "low"},
            "final_output": {
                "goal": "Async build",
                "total_steps": 1,
                "completed_steps": 1,
                "failed_steps": 0,
                "step_outputs": [],
                "overall_status": "completed",
            },
            "elapsed_seconds": 0.8,
        }

        def _submit_now(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return MagicMock()

        with patch("api.routes.Orchestrator") as MockOrch, patch("api.routes._EXECUTOR.submit", side_effect=_submit_now):
            MockOrch.return_value.run.return_value = mock_result
            submit = client.post("/run-task-async", json={"goal": "Build async pipeline"})
            assert submit.status_code == 200
            job_id = submit.json()["job_id"]

            status = client.get(f"/run-task-async/{job_id}")
            assert status.status_code == 200
            payload = status.json()
            assert payload["status"] == "completed"
            assert payload["result"]["goal"] == "Async build"

    def test_run_task_async_status_survives_routes_reload(self, client):
        import api.routes as routes

        def _submit_noop(fn, *args, **kwargs):
            # Simulate queued job persistence without immediate worker execution.
            return MagicMock()

        with patch("api.routes._EXECUTOR.submit", side_effect=_submit_noop):
            submit = client.post("/run-task-async", json={"goal": "Persist me"})
            assert submit.status_code == 200
            job_id = submit.json()["job_id"]

        reloaded_routes = importlib.reload(routes)
        job_status = reloaded_routes.run_task_status(job_id=job_id, _auth=None)

        assert job_status.job_id == job_id
        assert job_status.status == "pending"
        assert job_status.goal == "Persist me"

    def test_run_task_async_cancel_pending_job(self, client):
        with patch("api.routes._EXECUTOR.submit", return_value=MagicMock()):
            submit = client.post("/run-task-async", json={"goal": "Cancel me"})
            assert submit.status_code == 200
            job_id = submit.json()["job_id"]

        cancel = client.post(f"/run-task-async/{job_id}/cancel")
        assert cancel.status_code == 200
        payload = cancel.json()
        assert payload["status"] == "cancelled"

        polled = client.get(f"/run-task-async/{job_id}")
        assert polled.status_code == 200
        assert polled.json()["status"] == "cancelled"

    def test_run_task_async_cancel_running_job_returns_cancelling(self, client):
        from core.memory import create_async_job, set_async_job_state

        create_async_job(job_id="job-running", goal="Running", model="mistral", status="pending")
        set_async_job_state("job-running", status="running")

        cancel = client.post("/run-task-async/job-running/cancel")
        assert cancel.status_code == 200
        payload = cancel.json()
        assert payload["status"] == "cancelling"

    def test_run_task_async_cancel_completed_job_conflict(self, client):
        from core.memory import create_async_job, set_async_job_state

        create_async_job(job_id="job-complete", goal="Done", model="mistral", status="pending")
        set_async_job_state("job-complete", status="completed", result={"ok": True}, error=None)

        cancel = client.post("/run-task-async/job-complete/cancel")
        assert cancel.status_code == 409

    def test_run_task_async_cleanup_endpoint_applies_retention_query(self, client):
        import core.memory as mem
        from core.memory import create_async_job, get_async_job, set_async_job_state

        create_async_job(job_id="job-old", goal="Old", model="mistral", status="pending")
        set_async_job_state("job-old", status="completed", result={"ok": True}, error=None)

        with sqlite3.connect(mem.DB_PATH) as conn:
            conn.execute("UPDATE async_jobs SET updated_at = ? WHERE job_id = ?", (0.0, "job-old"))
            conn.commit()

        cleanup = client.post("/run-task-async/cleanup?retention_hours=1")
        assert cleanup.status_code == 200
        payload = cleanup.json()
        assert payload["retention_hours"] == 1
        assert payload["removed"] >= 1
        assert get_async_job("job-old") is None

    def test_get_task_not_found(self, client):
        resp = client.get("/task/99999")
        assert resp.status_code == 404

    def test_get_task_found(self, client):
        from core.memory import create_task

        task_id = create_task("Test goal for API")
        resp = client.get(f"/task/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["goal"] == "Test goal for API"

    def test_stream_task_events_returns_completed_job(self, client):
        from core.memory import create_async_job, set_async_job_state

        create_async_job(job_id="sse-1", goal="Stream test", model="mistral", status="pending")
        set_async_job_state(
            "sse-1",
            status="completed",
            result={"output": "done"},
            error=None,
        )
        resp = client.get("/run-task-async/sse-1/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_task_events_unknown_job(self, client):
        resp = client.get("/run-task-async/does-not-exist/stream")
        assert resp.status_code == 200
        body = resp.text
        assert "not found" in body

    def test_eval_endpoint_returns_metrics(self, client):
        from unittest.mock import patch

        mock_result = {
            "task_id": 1,
            "goal": "Eval test",
            "status": "completed",
            "plan": {"steps": ["step1"], "estimated_complexity": "low"},
            "final_output": {"result": "ok"},
            "step_results": [],
        }
        with patch("eval_harness.run_evaluation") as mock_eval:
            mock_eval.return_value = {
                "goal": "Eval test",
                "model": "mistral",
                "status": "completed",
                "elapsed_seconds": 1.23,
                "step_count": 1,
                "estimated_complexity": "low",
                "final_output_present": True,
                "step_results_count": 0,
            }
            resp = client.post("/eval?goal=Eval+test+goal&model=mistral")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            assert data["step_count"] == 1

    def test_metrics_endpoint_returns_counters(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "route_counts" in data
