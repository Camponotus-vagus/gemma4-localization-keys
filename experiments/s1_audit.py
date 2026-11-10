"""Audit of the provided code graphs, node embeddings and graph tools against gold-patch ground truth.

Usage: uv run python experiments/s1_audit.py
Needs data/derived/groundtruth.jsonl (experiments/s1_groundtruth.py).
Writes results/s1_audit/{summary.json,per_task.jsonl,report.md}.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from swegemma import graph as sg

from g4a.groundtruth import graph_path_for

DATA = Path("data/competition")
GT = Path("data/derived/groundtruth.jsonl")
OUT = Path("results/s1_audit")
RNG = np.random.default_rng(0)
K = 10

IDENT_RE = re.compile(r"`([^`\n]{2,80})`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b")


def issue_queries(text: str, limit: int = 12) -> list[str]:
    """Identifiers an agent would plausibly pass to search_similar_code: backticked spans and code-like tokens."""
    seen: list[str] = []
    for m in IDENT_RE.finditer(text):
        q = (m.group(1) or m.group(2)).strip().strip("()")
        if q and q not in seen and not q.startswith("http"):
            seen.append(q)
        if len(seen) >= limit:
            break
    return seen


def embedding_geometry(npz_path: Path, sample: int = 600) -> dict:
    z = np.load(npz_path)
    keys = list(z.files)
    idx = RNG.choice(len(keys), size=min(sample, len(keys)), replace=False)
    X = np.stack([z[keys[i]] for i in idx]).astype(np.float64)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T
    off = S[~np.eye(len(S), dtype=bool)]
    np.fill_diagonal(S, -np.inf)
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
    ev = np.clip(ev, 0, None)
    # Share of the uncentered second moment captured by the mean direction: 1.0 = all vectors parallel.
    mean_dir = Xn.mean(axis=0)
    return {
        "n_nodes": len(keys),
        "mean_cos": float(off.mean()),
        "p05_cos": float(np.quantile(off, 0.05)),
        "mean_top1_cos": float(S.max(axis=1).mean()),
        "pc1_var_share": float(ev[0] / ev.sum()),
        "participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()),
        "mean_resultant_length": float(np.linalg.norm(mean_dir)),
    }


def graph_stats(g: dict) -> dict:
    ids = [n["id"] for n in g["nodes"]]
    deg = Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    test_like = sum(1 for i in ids if i.split(".")[0] in {"tests", "test", "testing"} or ".test_" in i or i.startswith("test_"))
    kinds = Counter()
    for n in g["nodes"]:
        t = n["text"].lstrip()
        while t.startswith("@") and "\n" in t:
            t = t.split("\n", 1)[1].lstrip()
        kinds["async def" if t.startswith("async def") else "def" if t.startswith("def") else "class" if t.startswith("class") else "other"] += 1
    return {
        "nodes": len(ids),
        "edges": len(g["edges"]),
        "edge_types": dict(Counter(e.get("type") for e in g["edges"])),
        "test_node_share": test_like / max(len(ids), 1),
        "isolated_share": sum(1 for i in ids if deg[i] == 0) / max(len(ids), 1),
        "async_nodes": kinds["async def"],
        "kinds": dict(kinds),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truths = [json.loads(line) for line in GT.read_text().splitlines()]
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}

    # A. Coverage of gold symbols by graph nodes
    by_kind: dict[str, list[bool]] = defaultdict(list)
    task_rows = []
    for t in truths:
        syms = t["symbols"]
        for s in syms:
            by_kind[s["kind"] if s["qualname"] != "<new file>" else "new_file"].append(s["in_graph"])
        task_rows.append({
            "instance_id": t["instance_id"],
            "repo": t["repo"],
            "n_symbols": len(syms),
            "n_in_graph": sum(s["in_graph"] for s in syms),
            "any_in_graph": any(s["in_graph"] for s in syms),
            "all_in_graph": bool(syms) and all(s["in_graph"] for s in syms),
        })
    coverage = {k: {"n": len(v), "in_graph": sum(v), "share": sum(v) / len(v)} for k, v in sorted(by_kind.items())}

    # B + C. Graph composition and embedding geometry, once per distinct graph file
    graph_files = sorted({graph_path_for(tasks[t["instance_id"]], DATA / "graphs") for t in truths})
    per_graph = {}
    for gp in graph_files:
        g = json.loads(gp.read_text())
        per_graph[gp.stem] = {**graph_stats(g), **embedding_geometry(DATA / "embeddings" / f"{gp.stem}.npz")}

    # D. search_similar_code driven by identifiers from the issue text (as the harness implements it)
    for row in task_rows:
        task = tasks[row["instance_id"]]
        truth = next(t for t in truths if t["instance_id"] == row["instance_id"])
        gold = {s["node_id"] for s in truth["symbols"] if s["node_id"]}
        queries = issue_queries(task["problem_statement"])
        graph_key = graph_path_for(task, DATA / "graphs").stem
        returned: set[str] = set()
        resolved = 0
        direct_hit = False
        for q in queries:
            res = sg.get_similar_nodes(
                node=q, repo_name=graph_key, k=K,
                graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"),
            )
            if res:
                resolved += 1
            returned.update(r["node_name"] for r in res)
            g_obj = sg.get_graph(graph_key, graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"))
            if sg.resolve_node_name(q, g_obj) in gold:
                direct_hit = True
        row.update({
            "n_queries": len(queries),
            "queries_resolved": resolved,
            "tool_hit_at_10": bool(gold & returned),
            "name_resolution_hit": direct_hit,
            "gold_nodes": len(gold),
        })

    with_gold = [r for r in task_rows if r["gold_nodes"] > 0]
    summary = {
        "n_tasks": len(task_rows),
        "coverage_by_symbol_kind": coverage,
        "tasks_with_any_gold_node": sum(r["any_in_graph"] for r in task_rows),
        "tasks_with_all_gold_nodes": sum(r["all_in_graph"] for r in task_rows),
        "graphs": {
            "n": len(per_graph),
            **{k: float(np.mean([v[k] for v in per_graph.values()])) for k in (
                "nodes", "edges", "test_node_share", "isolated_share", "async_nodes",
                "mean_cos", "p05_cos", "mean_top1_cos", "pc1_var_share", "participation_ratio", "mean_resultant_length",
            )},
            "edge_types": dict(sum((Counter(v["edge_types"]) for v in per_graph.values()), Counter())),
        },
        "search_similar_code": {
            "tasks_evaluated": len(with_gold),
            "mean_queries_per_task": float(np.mean([r["n_queries"] for r in with_gold])),
            "share_queries_resolved": float(sum(r["queries_resolved"] for r in with_gold) / max(sum(r["n_queries"] for r in with_gold), 1)),
            "tasks_with_no_resolved_query": sum(r["queries_resolved"] == 0 for r in with_gold),
            "hit_at_10_union": sum(r["tool_hit_at_10"] for r in with_gold),
            "name_resolution_hit": sum(r["name_resolution_hit"] for r in with_gold),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in task_rows))
    (OUT / "per_graph.json").write_text(json.dumps(per_graph, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
