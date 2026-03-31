"""
tests/test_agents.py – Unit tests for all agents and core modules.

Tests mock the Ollama API so the suite runs fully offline without a
running Ollama instance.
"""

from __future__ import annotations

import json
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

    def test_safe_shell_blocked_command(self):
        from utils.helpers import safe_shell

        result = safe_shell(["rm", "-rf", "/"])
        assert result["return_code"] != 0
        assert "whitelist" in result["stderr"]

    def test_safe_shell_allowed_command(self):
        from utils.helpers import safe_shell

        result = safe_shell(["echo", "hello"])
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]

    def test_safe_shell_empty_command(self):
        from utils.helpers import safe_shell

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
        import api.routes as routes

        mem.DB_PATH = tmp_path / "test.db"
        mem.init_db()
        routes._JOBS.clear()

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
