"""S4: offline check of the repo_localization skill script against train snapshots.

Usage: uv run python experiments/s4_skill_check.py [--limit N] [--keep]
Extracts each snapshot to a temp dir under data/derived/, runs scripts/localize.py as a
subprocess (cwd = a different temp dir, like the skill runtime) with the problem statement
as query, parses the output and scores it with the S2 gold definition.
Writes results/s4_skill_check/{per_task.jsonl,summary.json}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import importlib.util

from g4a.localize import recall_at_k
from g4a.repo_index import is_candidate_file

DATA = Path("data/competition")
ZIP = DATA / "gemma-4-developer-agent.zip"
SCRIPT = Path("submission/skills_src/repo-localization/scripts/localize.py").resolve()
OUT = Path("results/s4_skill_check")
KS = (1, 5, 10)
UNIT_RE = re.compile(r"^\s*\d+\.\s+(\S+?):(\d+)-(\d+)\s{2}(\S+)\s{2}\(", re.M)
FILE_SECTION_RE = re.compile(r"^Top \d+ files:\n((?:  \S.*\n)+)", re.M)


def _load_script():
    spec = importlib.util.spec_from_file_location("localize_skill", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract(zf: zipfile.ZipFile, task: dict, dest: Path) -> None:
    local = DATA / "snapshots" / f"{task['instance_id']}.tgz"
    fh = open(local, "rb") if local.exists() else zf.open(f"snapshots/{task['instance_id']}.tgz")
    with fh, tarfile.open(fileobj=fh, mode="r|gz") as tf:
        tf.extractall(dest, filter="data")


SKILL = _load_script()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=20, help="--top passed to the script (unit@20 needs 20)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()]
    truths = {json.loads(l)["instance_id"]: json.loads(l) for l in Path("data/derived/groundtruth.jsonl").read_text().splitlines()}
    if args.limit:
        tasks = tasks[: args.limit]
    work = Path(tempfile.mkdtemp(prefix="s4_", dir=Path("data/derived").resolve()))
    cwd = Path(tempfile.mkdtemp(prefix="cwd_", dir=work))
    rows = []
    try:
        with zipfile.ZipFile(ZIP) as zf:
            for n, task in enumerate(tasks, 1):
                tid = task["instance_id"]
                repo_dir = work / "repo"
                shutil.rmtree(repo_dir, ignore_errors=True)
                repo_dir.mkdir()
                extract(zf, task, repo_dir)
                t0 = time.perf_counter()
                p = subprocess.run(
                    [sys.executable, str(SCRIPT), "--root", str(repo_dir), "--top", str(args.top),
                     "--query", task["problem_statement"]],
                    capture_output=True, text=True, cwd=cwd, timeout=120)
                dt = time.perf_counter() - t0
                units = [f"{m.group(1)}::{m.group(4)}" for m in UNIT_RE.finditer(p.stdout)]
                fm = FILE_SECTION_RE.search(p.stdout)
                files = [l.strip() for l in fm.group(1).splitlines()] if fm else []
                truth = truths[tid]
                present = {str(q.relative_to(repo_dir)) for q in repo_dir.rglob("*.py")}
                gold_files = {f for f in truth["files"] if f in present}
                # Same gold definition as S2: gold units must exist as indexed units at base_commit.
                skill_units, _, _ = SKILL.collect(str(repo_dir), time.monotonic() + 600)
                indexed = {f"{u.path}::{u.qualname}" for u in skill_units}
                gold_units = {f"{s['path']}::{s['qualname']}" for s in truth["symbols"]
                              if s["kind"] != "module"} & indexed
                row = {"instance_id": tid, "seconds": round(dt, 2), "rc": p.returncode,
                       "n_gold_files": len(gold_files), "n_gold_units": len(gold_units),
                       "stdout_chars": len(p.stdout)}
                if p.returncode:
                    row["stderr"] = p.stderr[-300:]
                for k in KS:
                    row[f"file_hit@{k}"] = recall_at_k(files, gold_files, k)[1]
                for k in (1, 5, 10, 20):
                    row[f"unit_hit@{k}"] = recall_at_k(units, gold_units, k)[1]
                row["n_units_parsed"] = len(units)
                rows.append(row)
                if n % 10 == 0:
                    print(f"[{n}/{len(tasks)}] {dt:.1f}s", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary: dict[str, float] = {}
    for key in [f"file_hit@{k}" for k in KS] + [f"unit_hit@{k}" for k in (1, 5, 10, 20)]:
        pool = [r for r in rows if r["n_gold_files" if key.startswith("file") else "n_gold_units"]]
        summary[key] = round(sum(r[key] for r in pool) / len(pool), 3)
        summary["n_" + key.split("_")[0]] = len(pool)
    secs = [r["seconds"] for r in rows]
    summary.update(median_s=round(statistics.median(secs), 2), max_s=max(secs),
                   max_chars=max(r["stdout_chars"] for r in rows), failures=sum(1 for r in rows if r["rc"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
