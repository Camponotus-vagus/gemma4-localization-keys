"""Driver for the g4a-repro notebook (embedded in the notebook by scripts/build_repro_notebook.py).

Runs the no-LLM external-set computations from public data only and compares them with the stored
reference numbers (results/s3_external, results/s3_graphkey/external) that ship with the notebook.
Expects the working directory to hold src/g4a, experiments/ and reference/ (the notebook writes them).
"""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REF = Path("reference")
CHECKS: list[tuple[str, bool, str]] = []

# Gold symbols by kind on the 153 tasks (paper/notes/s3_external.md, "Ground truth").
EXPECTED_KINDS = {"function": 263, "class": 30, "async_function": 0, "module": 44, "new_file": 1}
FLAT_CARDS, KEY_FILES = 20, 10
CEIL_KS = (20, 22, 25, 30, 40, 50)


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)


def ensure_deps() -> None:
    import importlib

    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("huggingface_hub", "huggingface_hub")):
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def restrict_repos(repos: list[str] | None) -> None:
    """Limit external.load_tasks() to some repos (used for the local dry run)."""
    from g4a import external as ext

    if not repos:
        return
    orig = ext.load_tasks
    ext.load_tasks = lambda: [t for t in orig() if t["repo"] in repos]  # type: ignore[assignment]


def tasks_in_scope() -> list[dict]:
    from g4a import external as ext

    return ext.load_tasks()


# ---------------------------------------------------------------- data

def download_dataset() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from g4a import external as ext

    if ext.DATASET.exists():
        print("dataset already present:", ext.DATASET)
        return
    path = hf_hub_download("princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    ext.EXT.mkdir(parents=True, exist_ok=True)
    ext.DATASET.write_text("".join(json.dumps(r) + "\n" for r in df.to_dict("records")))
    print(f"SWE-bench Verified test split: {len(df)} rows -> {ext.DATASET}")


def verify_selection() -> None:
    """The saved 153-task list must be what the seeded sampler gives on the freshly downloaded dataset."""
    from g4a import external as ext

    saved = json.loads(ext.TASKS_JSON.read_text())["instance_ids"]
    fresh = ext.select_ids()
    check("task selection reproduces (seed 20261024)", saved == fresh, f"{len(fresh)} ids")


def clone_repos(repos: list[str]) -> None:
    from g4a import external as ext

    ext.REPOS.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        bare = ext.bare_repo(repo)
        if bare.exists():
            print("have", bare)
            continue
        t0 = time.perf_counter()
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--bare", f"https://github.com/{repo}.git", str(bare)], check=True)
        print(f"cloned {repo} in {time.perf_counter() - t0:.0f}s", flush=True)


def build_snapshots(tasks: list[dict], workers: int = 4) -> None:
    from g4a import external as ext

    t0 = time.perf_counter()
    done = 0

    def one(t: dict) -> tuple[str, float]:
        ext.ensure_snapshot(t)
        return t["instance_id"], ext.snapshot_path(t).stat().st_size / 1e6

    with ThreadPoolExecutor(workers) as pool:
        for tid, mb in pool.map(one, tasks):
            done += 1
            print(f"[{done}/{len(tasks)}] {tid} {mb:.1f} MB ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- ground truth

def run_truth(tasks: list[dict]) -> None:
    from experiments import s3_external as s3
    from g4a import external as ext

    s3.cmd_truth(None)
    rows = s3.load_jsonl(ext.TRUTH_PATH)
    ref = {r["instance_id"]: r for r in map(json.loads, (REF / "groundtruth.jsonl").read_text().splitlines())}
    bad = [r["instance_id"] for r in rows if r != ref.get(r["instance_id"])]
    check("ground truth rows identical to the reference", not bad, f"{len(rows)} tasks, {len(bad)} differ {bad[:3]}")
    counts = dict(ext.symbol_kind_counts(rows))
    print("symbol kinds:", counts)
    if len(rows) == 153:
        check("symbol-kind table equals the paper (263/30/0/44/1)", all(counts.get(k, 0) == v for k, v in EXPECTED_KINDS.items()) and sum(counts.values()) == 338)


# ---------------------------------------------------------------- baselines

def compare_per_task(rows: list[dict], label: str) -> None:
    ref = {r["instance_id"]: r for r in map(json.loads, (REF / "per_task_s3_external.jsonl").read_text().splitlines())}
    n_cmp = n_bad = 0
    bad_ids = set()
    for r in rows:
        rr = ref[r["instance_id"]]
        for k, v in r.items():
            n_cmp += 1
            w = rr.get(k)
            same = (v == w) or (isinstance(v, float) and isinstance(w, float) and (math.isnan(v) and math.isnan(w) or abs(v - w) < 1e-9))
            if not same:
                n_bad += 1
                bad_ids.add(r["instance_id"])
    check(f"{label}: per-task values identical to results/s3_external", n_bad == 0,
          f"{len(rows)} tasks, {n_cmp} values compared, {n_bad} differ in {len(bad_ids)} tasks")


def run_baselines(methods: list[str]) -> list[dict]:
    from argparse import Namespace

    from experiments import s3_external as s3

    s3.cmd_baselines(Namespace(methods=methods))
    s3.cmd_baseline_summary(None)
    rows = s3.load_jsonl(s3.OUT / "per_task.jsonl")
    compare_per_task(rows, "BM25 files/units + PPR")
    return rows


def compare_summary(path: Path, methods: list[str], tol: float = 0.0) -> None:
    got = json.loads(path.read_text())
    ref = json.loads((REF / "summary_s3_external.json").read_text())
    diffs = [(m, k, got[m][k], ref[m][k]) for m in methods for k in got[m] if k in ref[m] and abs(got[m][k] - ref[m][k]) > tol]
    check("overall summary table equals results/s3_external/summary.json", not diffs, f"{len(diffs)} differences {diffs[:3]}")


# ---------------------------------------------------------------- BM25 candidate ceilings of the identification keys

def ceilings(tasks: list[dict]) -> dict:
    from experiments import s3_external as s3
    from g4a import external as ext
    from g4a.localize import rank_units_bm25

    truths = {r["instance_id"]: r for r in s3.load_jsonl(ext.TRUTH_PATH)}
    per_repo: dict[str, Counter] = defaultdict(Counter)
    tot: Counter = Counter()
    for t in tasks:
        idx = s3.load_index(t)
        gf, gu = s3.gold_for(truths[t["instance_id"]], idx)
        if not gu:
            continue
        rk = rank_units_bm25(idx, t["problem_statement"])[0]
        files: list[str] = []
        for uid in rk:
            p = uid.split("::", 1)[0]
            if p not in files:
                files.append(p)
            if len(files) == KEY_FILES:
                break
        c = per_repo[t["repo"]]
        for cc in (c, tot):
            cc["n"] += 1
            cc["flat_missing"] += not (set(rk[:FLAT_CARDS]) & gu)
            cc["dich_file_missing"] += not (set(files) & gf)
            for k in CEIL_KS:
                cc[f"unit_in_top{k}"] += bool(set(rk[:k]) & gu)
    n = tot["n"]
    print(f"\nBM25 candidate ceilings of the keys (tasks with a gold unit: {n})")
    print(f"{'repo':<28}{'n':>4}{'gold unit missing from 20 cards':>34}{'gold file missing from 10 files':>34}")
    for repo, c in sorted(per_repo.items()):
        print(f"{repo:<28}{c['n']:>4}{c['flat_missing']:>34}{c['dich_file_missing']:>34}")
    print(f"{'all':<28}{n:>4}{tot['flat_missing']:>34}{tot['dich_file_missing']:>34}")
    print(f"flat-key ceiling (unit in top {FLAT_CARDS}) = {1 - tot['flat_missing'] / n:.3f}")
    print("unit ceiling by number of cards:", {k: f"{tot[f'unit_in_top{k}']}/{n}={tot[f'unit_in_top{k}'] / n:.3f}" for k in CEIL_KS})
    if n == 151:
        diag = json.loads((REF / "diag_s3_external.json").read_text())["external"]["bm25"]
        check("flat ceiling: gold unit missing from 20 BM25 cards (54, .642)",
              tot["flat_missing"] == diag["flat_ceiling_missing"], f"got {tot['flat_missing']}, reference {diag['flat_ceiling_missing']}")
        check("dichotomous: gold file missing from 10 BM25 files (18)",
              tot["dich_file_missing"] == diag["dich_file_missing"], f"got {tot['dich_file_missing']}, reference {diag['dich_file_missing']}")
        gk = json.loads((REF / "graphkey_bm25_ceilings.json").read_text())
        bad = {k: (tot[f"unit_in_top{k}"], gk[f"bm25_k{k}"]["cand_unit_count"]) for k in CEIL_KS if tot[f"unit_in_top{k}"] != gk[f"bm25_k{k}"]["cand_unit_count"]}
        check("BM25 unit ceilings at 20/22/25/30/40/50 cards equal results/s3_graphkey/external", not bad, str(bad) if bad else "")
    return {"all": dict(tot), "per_repo": {r: dict(c) for r, c in per_repo.items()}}


# ---------------------------------------------------------------- repaired graph builder

def _kind_label(s: dict) -> str:
    if s["qualname"] == "<new file>":
        return "new_file"
    if s["kind"] == "module":
        return "module_level"
    return {"function": "sync_function", "async_function": "async_function", "class": "class"}[s["kind"]]


def graph_stats(tasks: list[dict], keep: Path | None = None) -> dict:
    """Build the repaired graph at each base commit (tasks with a gold unit) and report size and gold-symbol coverage."""
    from experiments import s3_external as s3
    from g4a import external as ext
    from g4a.graph_builder import build_graph_from_snapshot

    truths = {r["instance_id"]: r for r in s3.load_jsonl(ext.TRUTH_PATH)}
    per_repo: dict[str, dict] = defaultdict(lambda: {"graphs": 0, "nodes": 0, "edges": 0, "secs": 0.0, "edge_types": Counter(),
                                                     "cov": Counter(), "tot": Counter()})
    seen: dict[str, set] = {}  # commit key -> {(path, qualname)} of its graph nodes
    for t in tasks:
        idx = s3.load_index(t)
        if not s3.gold_for(truths[t["instance_id"]], idx)[1]:
            continue
        key = f"{t['repo'].split('/')[1]}_{t['base_commit']}"
        r = per_repo[t["repo"]]
        if key not in seen:
            t0 = time.perf_counter()
            g = build_graph_from_snapshot(ext.snapshot_path(t), key)
            r["secs"] += time.perf_counter() - t0
            r["graphs"] += 1
            r["nodes"] += len(g["nodes"])
            r["edges"] += len(g["edges"])
            r["edge_types"].update(e["type"] for e in g["edges"])
            seen[key] = {(n["path"], n["qualname"]) for n in g["nodes"]}
            if keep:
                keep.mkdir(parents=True, exist_ok=True)
                (keep / f"{key}.json.gz").write_bytes(gzip.compress(json.dumps(g).encode()))
        by_pq = seen[key]
        for s in truths[t["instance_id"]]["symbols"]:
            label = _kind_label(s)
            r["tot"][label] += 1
            if label != "new_file" and (s["path"], "<module>" if label == "module_level" else s["qualname"]) in by_pq:
                r["cov"][label] += 1
    labels = ("sync_function", "class", "async_function", "module_level", "new_file")
    print(f"\nRepaired graph builder on the external snapshots ({sum(r['graphs'] for r in per_repo.values())} graphs)")
    print(f"{'repo':<28}{'graphs':>7}{'nodes':>9}{'edges':>9}{'s/graph':>9}")
    out = {}
    for repo, r in sorted(per_repo.items()):
        print(f"{repo:<28}{r['graphs']:>7}{r['nodes'] / r['graphs']:>9.0f}{r['edges'] / r['graphs']:>9.0f}{r['secs'] / r['graphs']:>9.2f}")
        out[repo] = {"graphs": r["graphs"], "mean_nodes": r["nodes"] / r["graphs"], "mean_edges": r["edges"] / r["graphs"],
                     "edge_types": dict(r["edge_types"]), "cov": dict(r["cov"]), "tot": dict(r["tot"])}
    tot_edges: Counter = Counter()
    cov: Counter = Counter()
    tot: Counter = Counter()
    for r in per_repo.values():
        tot_edges.update(r["edge_types"])
        cov.update(r["cov"])
        tot.update(r["tot"])
    print("edges by type (all graphs):", dict(tot_edges))
    print("gold symbols that are nodes of the repaired graph:")
    for lab in labels:
        print(f"  {lab:<15}{cov[lab]:>5} / {tot[lab]:<5}")
    out["_edge_types"] = dict(tot_edges)
    return out


# ---------------------------------------------------------------- optional: Qwen3-Embedding

def run_embedding() -> None:
    """Embedding and RRF baselines. Slow on CPU (hundreds of thousands of units); off by default in the notebook."""
    from argparse import Namespace

    import torch

    import g4a.embed_rank as er
    from experiments import s3_external as s3

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("embedding device:", device)
    er.CACHE_DIR = s3.EMB_CACHE
    ranker = er.EmbeddingRanker(device=device)
    if device != "cpu":
        ranker.model.half()
    s3.embedder = lambda: ranker  # type: ignore[assignment]
    s3.OUT = Path("results/s3_external_emb")  # keep the BM25/PPR tables intact
    s3.cmd_baselines(Namespace(methods=["bm25_units", "emb", "emb+bm25"]))
    ref = json.loads((REF / "summary_s3_external.json").read_text())
    got = json.loads((s3.OUT / "summary.json").read_text())
    print("\nmax abs difference to the reference summary (fp16/fp32 and device change the scores slightly):")
    for m in ("emb", "emb+bm25"):
        print(" ", m, max(abs(got[m][k] - ref[m][k]) for k in got[m] if k in ref[m]))


def final_report() -> bool:
    print("\n==== checks ====")
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
    ok = all(c[1] for c in CHECKS)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok

