"""S0 end-to-end smoke test: full swegemma pipeline with Gemma 31B served locally by MLX.

The model runs on this Mac (mlx_lm.server on :8080); sandbox containers run on a
remote Docker host (DOCKER_HOST must point at it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import litellm
from adk_submission import ModelRegistry
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

SCORED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
LOCAL_MODEL = "mlx-community/gemma-4-31b-it-4bit"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--task-ids", nargs="+", required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--max-time-minutes", type=float, default=30)
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    args = ap.parse_args()

    litellm.drop_params = True
    data = Path("data/competition")
    models = ModelRegistry()
    models.register(SCORED_MODEL, LiteLlm(model=f"openai/{LOCAL_MODEL}", api_base=args.api_base, api_key="EMPTY"))
    limits, constraints = build_submission_limits()

    config = EvalConfig(
        tasks_path=data / "tasks.jsonl",
        snapshots_dir=data / "snapshots",
        results_dir=args.results_dir,
        submission_dir=args.submission,
        models=models,
        sandbox="docker",
        wheels_dir=data / "wheels",
        graph_dir=str(data / "graphs"),
        embeddings_dir=str(data / "embeddings"),
        max_time_minutes=args.max_time_minutes,
        max_tool_calls=100,
        limits=limits,
        generation_constraints=constraints,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5,
        ),
        display_mode="single",
    )
    evaluator = Evaluator(config)
    by_id = {t.instance_id: t for t in load_tasks(config.tasks_path)}
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for tid in args.task_ids:
        res = await evaluator.evaluate_task(by_id[tid])
        row = {
            "instance_id": tid,
            "resolved": res.resolved,
            "exit_code": res.test_exit_code,
            "tool_calls": res.tool_calls,
            "llm_calls": res.total_llm_calls,
            "seconds": round(res.duration_seconds, 1),
            "patch_chars": len(res.agent_patch or ""),
            "error": res.error,
        }
        (args.results_dir / f"{tid}.patch").write_text(res.agent_patch or "")
        with (args.results_dir / "local_agent.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
