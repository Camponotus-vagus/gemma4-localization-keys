"""Build the public-data reproduction notebook paper/notebook/g4a-repro/.

The notebook embeds the needed modules of src/g4a, experiments/s3_external.py, the saved task list, the driver
scripts/repro_runner.py and a few reference result files (all derived from public SWE-bench Verified data only).
No competition data, graphs or embeddings are used.

Usage:
    uv run python scripts/build_repro_notebook.py [--kaggle-user francesco simone slug]
Then:
    uv run kaggle kernels push -p paper/notebook/g4a-repro
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/notebook/g4a-repro"
SLUG = "g4a-repro"

# embedded path (relative to the notebook working directory) -> source file in the repository
EMBED = {
    "src/g4a/__init__.py": "src/g4a/__init__.py",
    "src/g4a/groundtruth.py": "src/g4a/groundtruth.py",
    "src/g4a/repo_index.py": "src/g4a/repo_index.py",
    "src/g4a/localize.py": "src/g4a/localize.py",
    "src/g4a/external.py": "src/g4a/external.py",
    "src/g4a/graph_builder.py": "src/g4a/graph_builder.py",
    "src/g4a/embed_rank.py": "src/g4a/embed_rank.py",
    "experiments/__init__.py": "experiments/__init__.py",
    "experiments/s3_external.py": "experiments/s3_external.py",
    "experiments/s3_external_tasks.json": "experiments/s3_external_tasks.json",
    "g4a_repro.py": "scripts/repro_runner.py",
    "reference/groundtruth.jsonl": "data/external/groundtruth.jsonl",
    "reference/per_task_s3_external.jsonl": "results/s3_external/per_task.jsonl",
    "reference/summary_s3_external.json": "results/s3_external/summary.json",
    "reference/diag_s3_external.json": "results/s3_external/diag.json",
}


def payload() -> dict[str, str]:
    files = {dst: base64.b64encode(zlib.compress((ROOT / src).read_bytes(), 9)).decode() for dst, src in EMBED.items()}
    gk = json.loads((ROOT / "results/s3_graphkey/external/retrieval_summary.json").read_text())
    small = {k: {"cand_unit_count": v["cand_unit_count"], "cand_unit_ceiling": v["cand_unit_ceiling"]} for k, v in gk.items() if k.startswith("bm25_k")}
    files["reference/graphkey_bm25_ceilings.json"] = base64.b64encode(zlib.compress(json.dumps(small).encode(), 9)).decode()
    return files


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# Reproducing the external-set results without an LLM

This notebook reruns, from public data only, the parts of the paper's external benchmark that need no language model.
The data are the 500 tasks of `princeton-nlp/SWE-bench_Verified` and the public GitHub repositories behind them.

What it reproduces (153 tasks on 8 repositories, same code as the project repository):
- the task selection (seed 20261024) and the gold-symbol table by kind (function, class, module-level, new file);
- the localization baselines that use no model: BM25 over files, BM25 over units, and personalized PageRank over our call graph;
- the BM25 candidate ceilings of the identification keys (share of tasks whose gold unit is among the 20 cards of the flat key, gold file among the 10 files of the dichotomous key);
- statistics of the repaired graph builder (size, edge types, share of gold symbols that are graph nodes).

Every table is compared with the stored numbers from the project (`results/s3_external`, `results/s3_graphkey/external`), which are embedded as reference files. A summary of PASS/FAIL checks is printed at the end.

What it does not reproduce:
- the identification-key runs (flat and dichotomous), which need Gemma 4 31B on a local GPU or Apple Silicon (`python -m experiments.s3_external key`, paper Section 5);
- every train-set result (Sections 3, 4.3 and 5 on the 129 competition tasks), which needs the competition data and so cannot run in a notebook that declares no competition source;
- the Qwen3-Embedding and RRF rows, unless the flag below is switched on.

The notebook needs internet (Hugging Face and github.com) and no GPU. Expect tens of minutes, mostly for the partial clones and `git archive` of 153 commits.
"""

CONFIG = """
import os
# Flags. REPOS = None runs all 8 repositories (the full 153 tasks); a list such as ["pytest-dev/pytest"] runs a subset.
REPOS = None
RUN_GRAPH = True        # build the repaired graph for each task commit and print its statistics (a few minutes)
RUN_EMBEDDING = False   # Qwen3-Embedding-0.6B + RRF rows; off by default, see the markdown cell above
WORKERS = 4             # parallel git archive jobs
"""

SETUP = """
import base64, json, os, subprocess, sys, zlib
from pathlib import Path

WORK = Path(os.environ.get("G4A_REPRO_WORK", "/kaggle/working/g4a-repro"))
if os.environ.get("G4A_REPRO_REPOS"):
    REPOS = os.environ["G4A_REPRO_REPOS"].split(",")  # local dry runs only
if os.environ.get("G4A_REPRO_FAST"):
    RUN_GRAPH = RUN_EMBEDDING = False
FILES = json.loads(%(payload)r)
for rel, b64 in FILES.items():
    p = WORK / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(zlib.decompress(base64.b64decode(b64)))
os.chdir(WORK)
sys.path[:0] = [str(WORK / "src"), str(WORK)]
print("working directory:", WORK, "|", len(FILES), "embedded files")
print(subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip(), "| python", sys.version.split()[0])

import g4a_repro as R
R.ensure_deps()
R.restrict_repos(REPOS)
"""

STEPS = [
    ("md", "## 1. Dataset and task list\nDownloads the SWE-bench Verified test split (parquet) and checks that the seeded sampler gives the same 153 task ids as the project's saved list."),
    ("code", """
R.download_dataset()
if REPOS is None:
    R.verify_selection()
TASKS = R.tasks_in_scope()
repos = sorted({t["repo"] for t in TASKS})
print(len(TASKS), "tasks in scope;", len(repos), "repositories")
"""),
    ("md", "## 2. Repositories and snapshots\nPartial clones (`--filter=blob:none --bare`) of the GitHub repositories, then `git archive` of the `*.py` files at each task's base commit, exactly as `g4a.external.ensure_snapshot` does."),
    ("code", """
R.clone_repos(repos)
R.build_snapshots(TASKS, WORKERS)
"""),
    ("md", "## 3. Ground truth by symbol kind\nGold patch -> enclosing functions and classes in the pre-fix source (same code as the S1 ground truth, no provided graph). The paper's table: 263 sync functions or methods, 30 classes, 0 async, 44 module-level edits, 1 new file (338 symbols)."),
    ("code", "R.run_truth(TASKS)"),
    ("md", "## 4. Baselines without a model\nBM25 over files, BM25 over units and personalized PageRank (BM25 seeds, our call graph). Hit rate = share of tasks with a gold item in the top k. The query is the problem statement only. Per-task values are compared with `results/s3_external/per_task.jsonl`."),
    ("code", """
rows = R.run_baselines(["bm25_files", "bm25_units", "ppr"])
if REPOS is None:
    R.compare_summary(Path("results/s3_external/summary.json"), ["bm25_files", "bm25_units", "ppr"])
"""),
    ("md", "## 5. BM25 candidate ceilings of the identification keys\nThe flat key shows 20 cards and the dichotomous key 10 files. If the gold item is not among them, no model can pick it. Counts are over the tasks with a gold unit (151 of 153) and are compared with `results/s3_external/diag.json` and `results/s3_graphkey/external`."),
    ("code", "ceil = R.ceilings(TASKS)"),
    ("md", "## 6. Repaired graph builder (optional)\nBuilds the stdlib-only AST graph at each task commit and reports size, edge types and the share of gold symbols that exist as nodes (async and module-level code are the cases the provided graphs miss; none of the 153 gold sets contains async code). There is no stored reference for these numbers; they are printed for inspection."),
    ("code", """
if RUN_GRAPH:
    gstats = R.graph_stats(TASKS)
else:
    print("RUN_GRAPH is False, skipped")
"""),
    ("md", "## 7. Embedding and RRF rows (optional, off by default)\nQwen3-Embedding-0.6B must embed every function and class of 153 repository snapshots, several hundred thousand texts. On the project's Apple GPU this took tens of minutes; on the CPU of a Kaggle notebook it is far too slow, so `RUN_EMBEDDING` is `False`. Set it to `True` with a GPU session if you want the embedding and RRF rows; scores differ slightly by device and precision, so this cell reports the maximum difference to the stored table instead of an exact-match check."),
    ("code", """
if RUN_EMBEDDING:
    R.run_embedding()
else:
    print("RUN_EMBEDDING is False, skipped")
"""),
    ("md", "## 8. Verdict"),
    ("code", """
ok = R.final_report()
"""),
]


def notebook() -> dict:
    cells = [md(INTRO), code(CONFIG), code(SETUP % {"payload": json.dumps(payload())})]
    for kind, text in STEPS:
        cells.append(md(text) if kind == "md" else code(text))
    return {"cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-user", default="francescosimone")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nb = notebook()
    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.kaggle_user}/{SLUG}",
        "title": SLUG,
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    print(OUT, f"{(OUT / (SLUG + '.ipynb')).stat().st_size / 1e6:.2f} MB", len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
