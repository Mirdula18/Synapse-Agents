"""
eval_harness.py - Lightweight evaluation harness for Synapse Agents.

Runs a synthetic goal through the full pipeline and reports metrics.
Usage: python eval_harness.py [--model mistral] [--goal "Build a todo app"]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from core.orchestrator import Orchestrator


def run_evaluation(goal: str, model: str = "mistral") -> dict[str, Any]:
    """Run the pipeline and return evaluation metrics."""
    start = time.time()
    orch = Orchestrator(model=model, enable_reflection=True)
    result = orch.run(goal)
    elapsed = time.time() - start

    plan = result.get("plan", {})
    steps = plan.get("steps", [])

    return {
        "goal": goal,
        "model": model,
        "status": "completed",
        "elapsed_seconds": round(elapsed, 2),
        "step_count": len(steps),
        "estimated_complexity": plan.get("estimated_complexity", "unknown"),
        "final_output_present": result.get("final_output") is not None,
        "step_results_count": len(result.get("step_results", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synapse Agents evaluation harness")
    parser.add_argument("--model", default="mistral", help="Ollama model name")
    parser.add_argument("--goal", default="Build a simple calculator web app", help="Goal to evaluate")
    args = parser.parse_args()

    print(f"Running evaluation with model={args.model}...")
    metrics = run_evaluation(args.goal, model=args.model)
    print(json.dumps(metrics, indent=2))

    if not metrics["final_output_present"]:
        print("\nWARNING: No final output produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
