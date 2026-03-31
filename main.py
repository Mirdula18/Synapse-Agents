"""
main.py – Application entry point.

Usage:
    # Start the API server (default port 8000):
    python main.py

    # Run a task directly from the CLI (offline, no HTTP server needed):
    python main.py --cli --goal "Build me a portfolio website"

    # Interactive mode (approve plan before execution):
    python main.py --cli --interactive --goal "Set up a FastAPI project"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core.settings import SETTINGS
from utils.helpers import setup_logging

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Synapse Agents",
    description=(
        "A multi-agent AI system that works fully offline using a local LLM (Ollama). "
        "Agents: Planner, Researcher, Executor, Self-Reflection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")

# Serve frontend
_FRONTEND = Path(__file__).parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend() -> FileResponse:
        return FileResponse(str(_FRONTEND / "index.html"))


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _run_cli(goal: str, model: str, interactive: bool, no_reflect: bool) -> None:
    """Run a task through the orchestrator directly (no HTTP)."""
    from core.orchestrator import Orchestrator

    def on_progress(event: str, data: dict) -> None:
        if event == "planning_done":
            plan = data.get("plan", {})
            print(f"\n📋 Plan ({plan.get('estimated_complexity', '')} complexity):")
            for i, step in enumerate(plan.get("steps", []), 1):
                print(f"   {i}. {step}")
        elif event == "step_start":
            idx = data.get("step_index", 0) + 1
            total = data.get("total", "?")
            print(f"\n⚙️  Step {idx}/{total}: {data.get('step', '')[:80]}")
        elif event == "research_done":
            print("   🔍 Research complete")
        elif event == "execution_done":
            print("   ✅ Execution complete")
        elif event == "reflection_done":
            score = data.get("reflection", {}).get("quality_score", "?")
            print(f"   🔎 Reflection: quality_score={score}")
        elif event == "completed":
            print("\n🎉 Pipeline completed!")
        elif event == "failed":
            print(f"\n❌ Pipeline failed: {data.get('error', 'Unknown error')}")

    orch = Orchestrator(
        model=model,
        enable_reflection=not no_reflect,
        interactive=interactive,
        progress_callback=on_progress,
    )

    print(f"\n🚀 Synapse Agents – Goal: {goal}")
    result = orch.run(goal)
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Synapse Agents – multi-agent AI system")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no HTTP server)")
    parser.add_argument("--goal", type=str, help="Goal to execute (CLI mode only)")
    parser.add_argument("--model", type=str, default=SETTINGS.default_model, help="Ollama model name")
    parser.add_argument("--interactive", action="store_true", help="Approve plan before executing")
    parser.add_argument("--no-reflect", action="store_true", help="Disable reflection agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.cli:
        if not args.goal:
            print("ERROR: --goal is required in --cli mode", file=sys.stderr)
            sys.exit(1)
        _run_cli(
            goal=args.goal,
            model=args.model,
            interactive=args.interactive,
            no_reflect=args.no_reflect,
        )
    else:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level=args.log_level.lower(),
        )


if __name__ == "__main__":
    main()
