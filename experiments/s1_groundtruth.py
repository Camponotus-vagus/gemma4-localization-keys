"""Build localization ground truth for all tasks, streaming snapshots out of the competition zip.

Usage: uv run python experiments/s1_groundtruth.py
Output: data/derived/groundtruth.jsonl (one TaskTruth per line)
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from g4a.groundtruth import build_truth, graph_path_for

DATA = Path("data/competition")
ZIP = DATA / "gemma-4-developer-agent.zip"
OUT = Path("data/derived/groundtruth.jsonl")


def main() -> None:
    tasks = {json.loads(line)["instance_id"]: json.loads(line) for line in (DATA / "tasks.jsonl").read_text().splitlines()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    with zipfile.ZipFile(ZIP) as zf:
        for i, (tid, task) in enumerate(tasks.items(), 1):
            local = DATA / "snapshots" / f"{tid}.tgz"
            if local.exists():
                truth = build_truth(task, local, graph_path_for(task, DATA / "graphs"))
            else:
                with zf.open(f"snapshots/{tid}.tgz") as fh:
                    truth = build_truth(task, fh, graph_path_for(task, DATA / "graphs"))
            rows.append(asdict(truth))
            print(f"[{i}/{len(tasks)}] {tid}: {len(truth.symbols)} symbols, "
                  f"{sum(s['in_graph'] for s in truth.symbols)} in graph ({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
