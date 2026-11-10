"""External benchmark tasks (SWE-bench Verified subset) in the same shape as the competition tasks.

Snapshots are `git archive` tarballs of the *.py files at base_commit, so that
g4a.repo_index.build_index and g4a.groundtruth.read_from_snapshot work unchanged.
There is no provided code graph for these repos: ground truth carries `in_graph=False`
and no node ids (graph coverage is measured separately from our own builder).
"""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from g4a.groundtruth import TaskTruth, TEST_PATH_RE, enclosing_symbols, parse_patch, read_from_snapshot

EXT = Path("data/external")
DATASET = EXT / "swebench_verified.jsonl"
REPOS = EXT / "repos"
SNAPSHOTS = EXT / "snapshots"
TASKS_JSON = Path("experiments/s3_external_tasks.json")
TRUTH_PATH = EXT / "groundtruth.jsonl"
SEED = 20261024

# repo -> number of tasks to sample (None = all). psf/requests is a train repo and is excluded.
SELECTION: dict[str, int | None] = {
    "pytest-dev/pytest": None,
    "pydata/xarray": None,
    "pylint-dev/pylint": None,
    "astropy/astropy": None,
    "sphinx-doc/sphinx": 20,
    "scikit-learn/scikit-learn": 20,
    "matplotlib/matplotlib": 20,
    "django/django": 20,
}


def load_dataset() -> dict[str, dict]:
    return {r["instance_id"]: r for r in map(json.loads, DATASET.read_text().splitlines())}


def select_ids() -> list[str]:
    data = load_dataset()
    rng = random.Random(SEED)
    ids: list[str] = []
    for repo, n in SELECTION.items():
        pool = sorted(i for i, r in data.items() if r["repo"] == repo)
        ids += pool if n is None else sorted(rng.sample(pool, n))
    return ids


def load_tasks() -> list[dict]:
    """Selected tasks in the saved order, each with the dataset fields."""
    data = load_dataset()
    spec = json.loads(TASKS_JSON.read_text())
    return [data[i] for i in spec["instance_ids"]]


def bare_repo(repo: str) -> Path:
    return REPOS / f"{repo.split('/')[1]}.git"


def snapshot_path(task: dict) -> Path:
    return SNAPSHOTS / f"{task['instance_id']}.tar.gz"


def ensure_snapshot(task: dict) -> Path:
    """Archive the *.py files at base_commit (idempotent). Blobs are fetched lazily from the partial clone."""
    out = snapshot_path(task)
    if out.exists():
        return out
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    subprocess.run(
        ["git", f"--git-dir={bare_repo(task['repo'])}", "archive", "--format=tar.gz", "-o", str(tmp),
         task["base_commit"], "--", "*.py"],
        check=True,
    )
    tmp.rename(out)
    return out


def build_truth_external(task: dict) -> TaskTruth:
    """Same logic as g4a.groundtruth.build_truth, without a provided graph."""
    edits = parse_patch(task["patch"])
    py_edits = [e for e in edits if e.path.endswith(".py")]
    existing = [e.path for e in py_edits if not e.is_new_file]
    sources = read_from_snapshot(snapshot_path(task), existing) if existing else {}

    symbols: list[dict] = []
    for e in py_edits:
        src = None if e.is_new_file else sources.get(e.path)
        if src is None:
            e.is_new_file = True
            symbols.append({"path": e.path, "qualname": "<new file>", "kind": "module", "lineno": 0,
                            "end_lineno": 0, "node_id": None, "in_graph": False})
            continue
        for s in enclosing_symbols(src, e.path, e.old_lines):
            symbols.append({**asdict(s), "node_id": None, "in_graph": False})
    return TaskTruth(
        instance_id=task["instance_id"], repo=task["repo"], base_commit=task["base_commit"],
        files=[e.path for e in edits],
        test_like_files=[e.path for e in edits if TEST_PATH_RE.search(e.path)],
        new_files=[e.path for e in edits if e.is_new_file],
        symbols=symbols,
    )


def symbol_kind_counts(truths: list[dict]) -> Counter:
    """Gold symbol counts by kind, with the S1 categories (new files and module-level edits separate)."""
    c: Counter = Counter()
    for t in truths:
        for s in t["symbols"]:
            c["new_file" if s["qualname"] == "<new file>" else s["kind"]] += 1
    return c
