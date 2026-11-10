"""S0 sanity check: gold patches must pass and empty patches must fail.

Reuses swegemma's own two-container pipeline (Evaluator.evaluate_task) and only
replaces the agent phase, so the verification path is identical to scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from adk_eval_core.tracing import SessionTrace
from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

from g4a.wheels import declared_requirements, materialize, pyproject_from_snapshot, select_wheels


async def check(evaluator: Evaluator, task, mode: str) -> dict:
    patch = task.patch if mode == "gold" else ""

    async def fake_agent(*_args, **_kwargs) -> tuple[str, str | None, SessionTrace]:
        return patch, None, SessionTrace()

    evaluator._run_agent_sandbox = fake_agent  # type: ignore[method-assign]
    t0 = time.perf_counter()
    res = await evaluator.evaluate_task(task)
    log_dir = evaluator.config.results_dir / "test_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{task.instance_id}.{mode}.log").write_text(res.test_output or "")
    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "mode": mode,
        "resolved": res.resolved,
        "exit_code": res.test_exit_code,
        "error": res.error,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def use_pinned_wheels(evaluator: Evaluator, args: argparse.Namespace, tid: str) -> None:
    """Point the evaluator at a per-task wheel selection, with its own swegemma cache dir."""
    text = pyproject_from_snapshot(args.snapshots_dir / f"{tid}.tgz") or ""
    reqs = declared_requirements(text) if text else []
    selected = select_wheels(sorted(args.wheels_dir.glob("*.whl")), reqs)
    wheels_dir = materialize(selected, args.wheels_dir.parent / "wheels_pinned")
    evaluator.config.wheels_dir = wheels_dir
    # swegemma caches unpacked wheels under gettempdir() with a fixed name: isolate per selection.
    base = Path(tempfile.gettempdir() if tempfile.tempdir is None else tempfile.tempdir).parent
    tempfile.tempdir = str(base / f"pin_{wheels_dir.name}")
    Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--snapshots-dir", type=Path, required=True)
    ap.add_argument("--wheels-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--task-ids", nargs="+", help="default: all tasks")
    ap.add_argument("--modes", nargs="+", default=["gold", "empty"])
    ap.add_argument("--pin-aware", action="store_true",
                    help="per task, keep only wheels satisfying the repo's pyproject constraints")
    args = ap.parse_args()

    by_id = {t.instance_id: t for t in load_tasks(args.tasks)}
    config = EvalConfig(
        tasks_path=args.tasks,
        snapshots_dir=args.snapshots_dir,
        results_dir=args.results_dir,
        submission_dir=args.results_dir,
        models={},
        sandbox="docker",
        wheels_dir=args.wheels_dir,
    )
    evaluator = Evaluator(config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "gold_check.jsonl"

    # Resumable: skip (task, mode) pairs already recorded.
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["instance_id"], row["mode"]))

    with out.open("a") as fh:
        for tid in args.task_ids or list(by_id):
            for mode in args.modes:
                if (tid, mode) in done:
                    continue
                if args.pin_aware:
                    use_pinned_wheels(evaluator, args, tid)
                row = await check(evaluator, by_id[tid], mode)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
