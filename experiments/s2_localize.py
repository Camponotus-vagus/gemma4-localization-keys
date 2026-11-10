"""S2: localization baselines against gold-patch ground truth.

Usage: uv run python experiments/s2_localize.py [--methods bm25_files bm25_units ppr provided_tool]
Needs data/derived/groundtruth.jsonl. Builds per-commit symbol indexes in data/derived/index/
(streaming snapshots from the competition zip when they are not extracted).
Writes results/s2_localize/{per_task.jsonl,summary.json} and prints a table.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from swegemma import graph as sg

from g4a.groundtruth import graph_path_for, module_candidates
from g4a.localize import files_from_units, rank_files_bm25, rank_units_bm25, rank_units_ppr, recall_at_k
from g4a.embed_rank import EmbeddingRanker, rrf
from g4a.repo_index import RepoIndex, build_index

DATA = Path("data/competition")
ZIP = DATA / "gemma-4-developer-agent.zip"
INDEX_DIR = Path("data/derived/index")
OUT = Path("results/s2_localize")
KS = (1, 5, 10, 20)


def load_index(task: dict, zf: zipfile.ZipFile) -> RepoIndex:
    key = f"{task['repo'].split('/')[1].lower()}_{task['base_commit']}"
    cache = INDEX_DIR / f"{key}.json.gz"
    if cache.exists():
        return RepoIndex.from_json(json.loads(gzip.decompress(cache.read_bytes())))
    local = DATA / "snapshots" / f"{task['instance_id']}.tgz"
    if local.exists():
        idx = build_index(local)
    else:
        with zf.open(f"snapshots/{task['instance_id']}.tgz") as fh:
            idx = build_index(fh)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(idx.to_json()).encode()))
    return idx


def provided_tool_ranking(task: dict, idx: RepoIndex, issue_queries) -> list[str]:
    """search_similar_code as the harness implements it, results mapped to our unit ids."""
    node_to_uid: dict[str, str] = {}
    for u in idx.units:
        for mod in module_candidates(u.path):
            node_to_uid.setdefault(f"{mod}.{u.qualname}", u.uid)
    key = graph_path_for(task, DATA / "graphs").stem
    ranking: list[str] = []
    for q in issue_queries(task["problem_statement"]):
        g = sg.get_graph(key, graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"))
        resolved = sg.resolve_node_name(q, g)
        hits = [resolved] if resolved else []
        hits += [r["node_name"] for r in sg.get_similar_nodes(
            node=q, repo_name=key, k=10, graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"))]
        for h in hits:
            uid = node_to_uid.get(h)
            if uid and uid not in ranking:
                ranking.append(uid)
    return ranking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["bm25_files", "bm25_units", "ppr", "emb", "emb+bm25", "provided_tool"])
    ap.add_argument("--with-hints", action="store_true", help="append hints_text to the issue")
    args = ap.parse_args()

    from experiments.s1_audit import issue_queries  # reuse the audit's query extraction

    embedder = EmbeddingRanker() if any(m.startswith("emb") for m in args.methods) else None

    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    truths = {json.loads(l)["instance_id"]: json.loads(l) for l in Path("data/derived/groundtruth.jsonl").read_text().splitlines()}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with zipfile.ZipFile(ZIP) as zf:
        for n, (tid, task) in enumerate(tasks.items(), 1):
            idx = load_index(task, zf)
            truth = truths[tid]
            gold_files = {p for p in truth["files"] if p in idx.files}
            uids = {u.uid for u in idx.units}
            gold_units = {f"{s['path']}::{s['qualname']}" for s in truth["symbols"]
                          if s["kind"] != "module"} & uids
            issue = task["problem_statement"] + ("\n" + task.get("hints_text", "") if args.with_hints else "")
            row = {"instance_id": tid, "repo": task["repo"], "n_gold_files": len(gold_files),
                   "n_gold_units": len(gold_units), "n_units": len(idx.units), "n_files": len(idx.files)}
            for m in args.methods:
                if m == "bm25_files":
                    files, units = rank_files_bm25(idx, issue), None
                elif m == "bm25_units":
                    units, _ = rank_units_bm25(idx, issue)
                    files = files_from_units(units)
                elif m == "ppr":
                    units = rank_units_ppr(idx, issue)
                    files = files_from_units(units)
                elif m == "emb":
                    units, _ = embedder.rank_units(idx, issue)
                    files = files_from_units(units)
                elif m == "emb+bm25":
                    units = rrf(embedder.rank_units(idx, issue)[0], rank_units_bm25(idx, issue)[0])
                    files = files_from_units(units)
                elif m == "provided_tool":
                    units = provided_tool_ranking(task, idx, issue_queries)
                    files = files_from_units(units)
                else:
                    raise SystemExit(f"unknown method {m}")
                for k in KS:
                    r, h = recall_at_k(files, gold_files, k)
                    row[f"{m}/file_recall@{k}"], row[f"{m}/file_hit@{k}"] = r, h
                    if units is not None:
                        r, h = recall_at_k(units, gold_units, k)
                        row[f"{m}/unit_recall@{k}"], row[f"{m}/unit_hit@{k}"] = r, h
            rows.append(row)
            if n % 10 == 0:
                print(f"[{n}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)

    (OUT / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary: dict[str, dict[str, float]] = defaultdict(dict)
    for key in sorted({k for r in rows for k in r if "/" in k}):
        vals = [r[key] for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))
                and (r["n_gold_units"] if "unit_" in key else r["n_gold_files"])]
        m, metric = key.split("/")
        summary[m][metric] = round(float(np.mean(vals)), 3) if vals else float("nan")
        summary[m][f"n_{'units' if 'unit_' in key else 'files'}"] = len(vals)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["file_hit@1", "file_hit@5", "file_hit@10", "unit_hit@1", "unit_hit@5", "unit_hit@10", "unit_hit@20", "unit_recall@10"]
    print(f"\n{'method':<14}" + "".join(f"{c:>15}" for c in cols))
    for m in args.methods:
        print(f"{m:<14}" + "".join(f"{summary[m].get(c, float('nan')):>15.3f}" for c in cols))
    print(f"\ntasks with gold files: {summary[args.methods[0]].get('n_files')}, with gold units: "
          f"{summary.get('bm25_units', {}).get('n_units')}")


if __name__ == "__main__":
    main()
