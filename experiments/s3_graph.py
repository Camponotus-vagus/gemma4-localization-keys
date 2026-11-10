"""S3a: repaired code graph vs the provided graphs, on the 129 train tasks.

Usage (from the project root, after `source scripts/env.sh`):
    uv run python -m experiments.s3_graph [build|analyze|localize|all]     (default: all)

  build     build repaired graphs from the snapshots -> data/derived/graphs_repaired/*.json.gz
  analyze   gold-symbol coverage by kind, graph composition, id/edge compatibility with the provided graphs
  localize  BM25-seeded personalized PageRank over the old call graph and the repaired graph
            (edge-type ablations), then a BM25 + PPR hybrid tuned with leave-one-repo-out

Needs data/derived/groundtruth.jsonl (experiments/s1_groundtruth.py) and the unit indexes of S2.
Writes results/s3_graph/{summary.json,per_task.jsonl}; stages merge into existing files.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from g4a.graph_builder import EDGE_TYPES, build_graph_from_snapshot, module_name_of
from g4a.groundtruth import graph_path_for
from g4a.localize import (call_graph, files_from_units, personalized_pagerank, rank_units_bm25, recall_at_k)

DATA = Path("data/competition")
ZIP = DATA / "gemma-4-developer-agent.zip"
GT = Path("data/derived/groundtruth.jsonl")
GRAPHS = Path("data/derived/graphs_repaired")
OUT = Path("results/s3_graph")
S2_PER_TASK = Path("results/s2_localize/per_task.jsonl")
KS = (1, 5, 10, 20)
SEMANTIC = ("calls", "inherits", "references", "maybe_calls")
POOL = 6

_TASKS: dict[str, dict] = {}
_TRUTH: dict[str, dict] = {}


def _init() -> None:
    global _TASKS, _TRUTH
    _TASKS = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    _TRUTH = {json.loads(l)["instance_id"]: json.loads(l) for l in GT.read_text().splitlines()}


def graph_key(task: dict) -> str:
    return f"{task['repo'].split('/')[1].lower()}_{task['base_commit']}"


def repaired_graph(task: dict) -> dict:
    """Load the cached repaired graph of a task, building it from the snapshot when missing."""
    cache = GRAPHS / f"{graph_key(task)}.json.gz"
    if cache.exists():
        return json.loads(gzip.decompress(cache.read_bytes()))
    local = DATA / "snapshots" / f"{task['instance_id']}.tgz"
    if local.exists():
        g = build_graph_from_snapshot(local, graph_key(task))
    else:
        with zipfile.ZipFile(ZIP) as zf, zf.open(f"snapshots/{task['instance_id']}.tgz") as fh:
            g = build_graph_from_snapshot(fh, graph_key(task))
    GRAPHS.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(g).encode()))
    return g


def _build_one(tid: str) -> tuple[str, float, int, int]:
    t0 = time.perf_counter()
    g = repaired_graph(_TASKS[tid])
    return tid, time.perf_counter() - t0, len(g["nodes"]), len(g["edges"])


# ---------------------------------------------------------------------------------------------
# analyze: coverage, composition, compatibility
# ---------------------------------------------------------------------------------------------

def _kind_label(s: dict) -> str:
    if s["qualname"] == "<new file>":
        return "new_file"
    if s["kind"] == "module":
        return "module_level"
    return {"function": "sync_function", "async_function": "async_function", "class": "class"}[s["kind"]]


def _provided_stats(g: dict) -> dict:
    ids = [n["id"] for n in g["nodes"]]
    deg: Counter[str] = Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    test = sum(1 for i in ids if i.split(".")[0] in {"tests", "test", "testing"} or ".test_" in i or i.startswith("test_"))
    return {"nodes": len(ids), "edges": len(g["edges"]), "isolated": sum(1 for i in ids if deg[i] == 0) / max(len(ids), 1),
            "test_share": test / max(len(ids), 1), "edge_types": dict(Counter(e.get("type") for e in g["edges"]))}


def _repaired_stats(g: dict) -> dict:
    kinds = Counter(n["kind"] for n in g["nodes"])
    deg_all: Counter[str] = Counter()
    deg_sem: Counter[str] = Counter()
    for e in g["edges"]:
        for end in (e["source"], e["target"]):
            deg_all[end] += 1
            if e["type"] in SEMANTIC:
                deg_sem[end] += 1
    fc = [n for n in g["nodes"] if n["kind"] != "module"]
    return {"nodes": len(g["nodes"]), "nodes_fc": len(fc), "edges": len(g["edges"]), "kinds": dict(kinds),
            "edge_types": dict(Counter(e["type"] for e in g["edges"])),
            "isolated_any": sum(1 for n in g["nodes"] if deg_all[n["id"]] == 0) / max(len(g["nodes"]), 1),
            "isolated_semantic_fc": sum(1 for n in fc if deg_sem[n["id"]] == 0) / max(len(fc), 1),
            "test_share_fc": sum(n["is_test"] for n in fc) / max(len(fc), 1),
            "test_share_all": sum(n["is_test"] for n in g["nodes"]) / max(len(g["nodes"]), 1)}


def _analyze_one(tid: str) -> dict:
    task, truth = _TASKS[tid], _TRUTH[tid]
    rg = repaired_graph(task)
    pg = json.loads(graph_path_for(task, DATA / "graphs").read_text())
    pids = {n["id"] for n in pg["nodes"]}
    by_pq = {(n["path"], n["qualname"]): n for n in rg["nodes"]}
    module_dirs = {n["path"].rsplit("/", 1)[0] if "/" in n["path"] else "" for n in rg["nodes"] if n["kind"] == "module"}

    cov: list[dict] = []
    for s in truth["symbols"]:
        label = _kind_label(s)
        if label == "new_file":
            node = None
            d = s["path"].rsplit("/", 1)[0] if "/" in s["path"] else ""
            anchor = d in module_dirs  # a module of the same package exists: an attach point for the new file
        else:
            node = by_pq.get((s["path"], "<module>" if label == "module_level" else s["qualname"]))
            anchor = False
        # Lenient test of the provided graph: legacy id (function names dropped) or class-less flat id.
        lenient = s["in_graph"]
        if node is not None and not lenient and label in ("sync_function", "class"):
            flat = f"{module_name_of(node['path'])[0]}.{node['qualname'].split('.')[-1]}"
            lenient = node["legacy_id"] in pids or flat in pids
        cov.append({"kind": label, "provided": bool(s["in_graph"]), "provided_lenient": bool(lenient),
                    "repaired": node is not None, "new_file_anchor": anchor})

    # Compatibility with the provided graph via legacy ids
    leg = {}
    for n in rg["nodes"]:
        if n["kind"] not in ("module", "async_function"):
            leg.setdefault(n["legacy_id"], n["id"])
    matched = [p for p in pids if p in leg]
    sem_pairs = {frozenset((e["source"], e["target"])) for e in rg["edges"] if e["type"] in SEMANTIC}
    calls_pairs = {frozenset((e["source"], e["target"])) for e in rg["edges"] if e["type"] == "calls"}
    prov_pairs = set()
    for e in pg["edges"]:
        a, b = leg.get(e["source"]), leg.get(e["target"])
        if a and b and a != b:
            prov_pairs.add(frozenset((a, b)))
    inv = set(leg.values())
    ours_in_prov = {p for p in calls_pairs if all(x in {leg[m] for m in matched} for x in p)}
    compat = {
        "provided_ids": len(pids), "provided_ids_found": len(matched),
        "repaired_fc_ids_not_in_provided": len({n["legacy_id"] for n in rg["nodes"] if n["kind"] not in ("module", "async_function")} - pids),
        "provided_edges_pairs": len(prov_pairs), "provided_pairs_in_repaired_semantic": len(prov_pairs & sem_pairs),
        "provided_pairs_in_repaired_calls": len(prov_pairs & calls_pairs),
        "repaired_calls_pairs_among_provided_nodes": len(ours_in_prov), "of_which_in_provided": len(ours_in_prov & prov_pairs),
    }
    return {"instance_id": tid, "repo": task["repo"], "key": graph_key(task), "coverage": cov,
            "provided": _provided_stats(pg), "repaired": _repaired_stats(rg), "compat": compat}


def analyze(pool: Pool, tids: list[str]) -> tuple[dict, dict[str, dict]]:
    res = pool.map(_analyze_one, tids)
    kinds = ["sync_function", "class", "async_function", "module_level", "new_file"]
    table: dict[str, dict] = {}
    for k in kinds:
        rows = [c for r in res for c in r["coverage"] if c["kind"] == k]
        table[k] = {"n": len(rows), "provided": sum(c["provided"] for c in rows),
                    "provided_lenient": sum(c["provided_lenient"] for c in rows),
                    "repaired": sum(c["repaired"] for c in rows)}
        if k == "new_file":
            table[k]["repaired_anchor_exists"] = sum(c["new_file_anchor"] for c in rows)

    def task_cover(r: dict, field: str) -> tuple[bool, bool]:
        c = [x[field] for x in r["coverage"]]
        return any(c), bool(c) and all(c)

    tasks = {f: [task_cover(r, f) for r in res] for f in ("provided", "provided_lenient", "repaired")}
    # one row per distinct graph
    seen: dict[str, dict] = {}
    for r in res:
        seen.setdefault(r["key"], r)
    prov = [r["provided"] for r in seen.values()]
    rep = [r["repaired"] for r in seen.values()]
    compat = [r["compat"] for r in seen.values()]

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([x[key] for x in rows]))

    def sum_types(rows: list[dict]) -> dict:
        return {k: float(np.mean([x["edge_types"].get(k, 0) for x in rows])) for k in sorted({k for x in rows for k in x["edge_types"]})}

    tot = lambda key: sum(c[key] for c in compat)  # noqa: E731
    summary = {
        "n_tasks": len(res), "n_graphs": len(seen),
        "coverage_by_kind": table,
        "tasks_with_any_gold_node": {f: sum(a for a, _ in v) for f, v in tasks.items()},
        "tasks_with_all_gold_nodes": {f: sum(b for _, b in v) for f, v in tasks.items()},
        "composition_provided": {"nodes": mean(prov, "nodes"), "edges": mean(prov, "edges"), "isolated": mean(prov, "isolated"),
                                 "test_share": mean(prov, "test_share"), "edge_types_mean": sum_types(prov)},
        "composition_repaired": {**{k: mean(rep, k) for k in ("nodes", "nodes_fc", "edges", "isolated_any", "isolated_semantic_fc",
                                                              "test_share_fc", "test_share_all")},
                                 "edge_types_mean": sum_types(rep),
                                 "node_kinds_mean": {k: float(np.mean([x["kinds"].get(k, 0) for x in rep])) for k in
                                                     ("module", "class", "function", "async_function")}},
        "compat": {
            "provided_ids_found_share": tot("provided_ids_found") / tot("provided_ids"),
            "provided_ids": tot("provided_ids"), "provided_ids_found": tot("provided_ids_found"),
            "repaired_fc_ids_not_in_provided": tot("repaired_fc_ids_not_in_provided"),
            "provided_edge_pairs": tot("provided_edges_pairs"),
            "provided_pairs_in_repaired_semantic_share": tot("provided_pairs_in_repaired_semantic") / max(tot("provided_edges_pairs"), 1),
            "provided_pairs_in_repaired_calls_share": tot("provided_pairs_in_repaired_calls") / max(tot("provided_edges_pairs"), 1),
            "repaired_calls_pairs_among_provided_nodes": tot("repaired_calls_pairs_among_provided_nodes"),
            "repaired_calls_pairs_in_provided_share": tot("of_which_in_provided") / max(tot("repaired_calls_pairs_among_provided_nodes"), 1),
        },
    }
    per_task = {r["instance_id"]: {"n_symbols": len(r["coverage"]),
                                   "cov_provided": sum(c["provided"] for c in r["coverage"]),
                                   "cov_repaired": sum(c["repaired"] for c in r["coverage"])} for r in res}
    return summary, per_task


# ---------------------------------------------------------------------------------------------
# localize: BM25-seeded PPR over graph variants
# ---------------------------------------------------------------------------------------------

def sym_edges(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected, deduplicated, loop-free edge arrays (both directions)."""
    keep = src != dst
    a, b = np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])
    key = np.unique(a.astype(np.int64) * n + b)
    a, b = key // n, key % n
    return np.concatenate([a, b]), np.concatenate([b, a])


def ppr(n: int, src: np.ndarray, dst: np.ndarray, seed: np.ndarray, alpha: float = 0.25, iters: int = 30) -> np.ndarray:
    """Personalized PageRank with restart alpha; dangling mass returns to the seed (as g4a.localize)."""
    p = seed / seed.sum() if seed.sum() > 0 else np.full(n, 1 / n)
    deg = np.bincount(src, minlength=n).astype(float)
    inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
    dangling = deg == 0
    r = p.copy()
    for _ in range(iters):
        nxt = np.bincount(dst, weights=(r * inv)[src], minlength=n)
        r = alpha * p + (1 - alpha) * (nxt + r[dangling].sum() * p)
    return r


EDGE_VARIANTS: dict[str, tuple[str, ...]] = {
    "rep_all": EDGE_TYPES,
    **{f"rep_minus_{t}": tuple(x for x in EDGE_TYPES if x != t) for t in EDGE_TYPES},
    "rep_calls_only": ("calls",),
    "rep_calls_contains": ("calls", "contains"),
    "rep_semantic": SEMANTIC,
}
SEED_GRID = (5, 10, 20, 50)


def _localize_one(tid: str) -> dict:
    from experiments.s2_localize import load_index  # same unit index as S2
    task, truth = _TASKS[tid], _TRUTH[tid]
    with zipfile.ZipFile(ZIP) as zf:
        idx = load_index(task, zf)
    g = repaired_graph(task)
    issue = task["problem_statement"]
    _, s = rank_units_bm25(idx, issue)
    units = idx.units
    nu = len(units)

    gold_files = sorted({p for p in truth["files"] if p in idx.files})
    uids = [u.uid for u in units]
    gold_units = sorted({f"{x['path']}::{x['qualname']}" for x in truth["symbols"] if x["kind"] != "module"} & set(uids))

    # Repaired graph: node index space; units map to the first node with the same (path, qualname).
    nid = {n["id"]: i for i, n in enumerate(g["nodes"])}
    pq = {}
    for i, n in enumerate(g["nodes"]):
        pq.setdefault((n["path"], n["qualname"]), i)
    unit_node = np.array([pq.get((u.path, u.qualname), -1) for u in units])
    is_test = np.array([n["is_test"] for n in g["nodes"]])
    N = len(g["nodes"])
    by_type = {t: (np.array([nid[e["source"]] for e in g["edges"] if e["type"] == t], dtype=np.int64),
                   np.array([nid[e["target"]] for e in g["edges"] if e["type"] == t], dtype=np.int64)) for t in EDGE_TYPES}

    def seed_vec(top: int, size: int, mapping: np.ndarray | None) -> np.ndarray:
        order = np.argsort(-s, kind="stable")[:top]
        v = np.zeros(size)
        for i in order:
            j = i if mapping is None else mapping[i]
            if j >= 0:
                v[j] = max(v[j], max(s[i], 0.0))
        return v

    def over_units(r: np.ndarray, mapping: np.ndarray | None) -> np.ndarray:
        if mapping is None:
            return r
        out = np.zeros(nu)
        ok = mapping >= 0
        out[ok] = r[mapping[ok]]
        return out

    vecs: dict[str, np.ndarray] = {}

    # Old approximate call graph (g4a.localize), same PPR as S2; seeds grid for the hybrid.
    adj = call_graph(idx)
    osrc = np.array([i for i, nb in adj.items() for _ in nb], dtype=np.int64)
    odst = np.array([j for i, nb in adj.items() for j in nb], dtype=np.int64)
    osrc, odst = sym_edges(nu, osrc, odst) if len(osrc) else (osrc, odst)
    for top in SEED_GRID:
        vecs[f"old_callgraph@{top}"] = ppr(nu, osrc, odst, seed_vec(top, nu, None))

    # Repaired graph variants (seeds = top 20 BM25 units), plus a seed grid for the full graph.
    def variant_vec(types: tuple[str, ...], top: int, drop_tests: bool = False) -> np.ndarray:
        src = np.concatenate([by_type[t][0] for t in types]) if types else np.array([], dtype=np.int64)
        dst = np.concatenate([by_type[t][1] for t in types]) if types else np.array([], dtype=np.int64)
        if drop_tests:
            keep = ~(is_test[src] | is_test[dst])
            src, dst = src[keep], dst[keep]
        a, b = sym_edges(N, src, dst)
        return over_units(ppr(N, a, b, seed_vec(top, N, unit_node)), unit_node)

    for name, types in EDGE_VARIANTS.items():
        vecs[f"{name}@20"] = variant_vec(types, 20)
    vecs["rep_all_no_tests@20"] = variant_vec(EDGE_TYPES, 20, drop_tests=True)
    for top in SEED_GRID:
        if top != 20:
            vecs[f"rep_all@{top}"] = variant_vec(EDGE_TYPES, top)

    # Reachability diagnostic: gold units outside the BM25 top 20 that sit 1 or 2 hops from a top-20 seed.
    top20 = np.argsort(-s, kind="stable")[:20]
    gold_idx = np.array([i for i, u in enumerate(uids) if u in set(gold_units)], dtype=np.int64)
    outside = np.setdiff1d(gold_idx, top20)

    def hops(a: np.ndarray, b: np.ndarray, size: int, seeds: np.ndarray, to_unit: np.ndarray | None) -> dict:
        m1 = np.zeros(size, dtype=bool)
        m1[b[np.isin(a, seeds)]] = True
        m2 = m1.copy()
        m2[b[np.isin(a, np.nonzero(m1)[0])]] = True
        out = {}
        for h, m in ((1, m1), (2, m2)):
            um = m if to_unit is None else np.array([(j >= 0 and m[j]) for j in to_unit])
            um[top20] = False  # candidates outside the top 20
            out[h] = (int(um.sum()), int(um[outside].sum()) if len(outside) else 0)
        return out

    reach = {"gold_outside_top20": int(len(outside)), "old_callgraph": hops(osrc, odst, nu, top20, None)}
    ssrc = np.concatenate([by_type[t][0] for t in SEMANTIC])
    sdst = np.concatenate([by_type[t][1] for t in SEMANTIC])
    a, b = sym_edges(N, ssrc, sdst)
    reach["rep_semantic"] = hops(a, b, N, unit_node[top20][unit_node[top20] >= 0], unit_node)

    return {"instance_id": tid, "repo": task["repo"], "reach": reach, "uids": uids, "bm25": s, "gold_files": gold_files, "gold_units": gold_units,
            "vecs": {k: v.astype(np.float32) for k, v in vecs.items()}, "n_units": nu,
            "unmapped_units": int((unit_node < 0).sum())}


def _rank(bm: np.ndarray, pr: np.ndarray | None, mix: float) -> np.ndarray:
    """Unit order for score = mix * bm/max + (1 - mix) * ppr/max (the S2 formula); pr None = plain BM25."""
    b = bm / bm.max() if bm.max() > 0 else bm
    if pr is None:
        return np.argsort(-b, kind="stable")
    p = pr / pr.max() if pr.max() > 0 else pr
    return np.argsort(-(mix * b + (1 - mix) * p), kind="stable")


def _metrics(order: np.ndarray, rec: dict) -> dict[str, float]:
    ranking = [rec["uids"][i] for i in order]
    files = files_from_units(ranking)
    out: dict[str, float] = {}
    gf, gu = set(rec["gold_files"]), set(rec["gold_units"])
    for k in KS:
        if gf:
            out[f"file_hit@{k}"] = float(recall_at_k(files, gf, k)[1])
        if gu:
            out[f"unit_hit@{k}"] = float(recall_at_k(ranking, gu, k)[1])
            out[f"unit_recall@{k}"] = float(recall_at_k(ranking, gu, k)[0])
    return out


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: round(float(np.mean([r[k] for r in rows if k in r])), 3) for k in keys} | \
           {"n_files": sum("file_hit@1" in r for r in rows), "n_units": sum("unit_hit@1" in r for r in rows)}


def localize(pool: Pool, tids: list[str]) -> tuple[dict, dict[str, dict]]:
    recs = pool.map(_localize_one, tids, chunksize=2)
    per_task: dict[str, dict] = {r["instance_id"]: {} for r in recs}
    method_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    method_rows_by_task: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    def add(method: str, rec: dict, order: np.ndarray) -> None:
        m = _metrics(order, rec)
        method_rows[method].append(m)
        method_rows_by_task[method][rec["instance_id"]] = m
        per_task[rec["instance_id"]].update({f"{method}/{k}": v for k, v in m.items()})

    for rec in recs:
        add("bm25_units", rec, _rank(rec["bm25"], None, 1.0))
        for name, vec in rec["vecs"].items():
            base, top = name.split("@")
            if int(top) != 20:
                continue
            v = vec.astype(float)
            add(f"{base}/mix0.5", rec, _rank(rec["bm25"], v, 0.5))
            add(f"{base}/pure", rec, _rank(rec["bm25"] * 1e-9, v, 0.0))  # ppr order, BM25 only breaks ties

    # Hybrid tuning: leave-one-repo-out over (mix, seeds); the held-out repo never influences its own parameters.
    grid_mix = (0.3, 0.5, 0.7, 0.85, 0.95)
    repos = sorted({r["repo"] for r in recs})

    def score(rows: list[dict[str, float]]) -> float:
        u = [r for r in rows if "unit_hit@1" in r]
        return float(np.mean([(r["unit_hit@1"] + r["unit_hit@5"] + r["unit_hit@10"]) / 3 for r in u])) if u else 0.0

    hybrid_summary: dict[str, dict] = {}
    for family in ("old_callgraph", "rep_all"):
        cache: dict[tuple[int, float], dict[str, dict[str, float]]] = {}
        for top in SEED_GRID:
            for mix in grid_mix:
                cache[(top, mix)] = {r["instance_id"]: _metrics(_rank(r["bm25"], r["vecs"][f"{family}@{top}"].astype(float), mix), r)
                                     for r in recs}
        chosen: dict[str, tuple[int, float]] = {}
        loro_rows: dict[str, dict[str, float]] = {}
        for repo in repos:
            train = [r["instance_id"] for r in recs if r["repo"] != repo]
            best = max(cache, key=lambda p: (score([cache[p][t] for t in train]), -abs(p[1] - 0.7)))
            chosen[repo] = best
            for r in recs:
                if r["repo"] == repo:
                    loro_rows[r["instance_id"]] = cache[best][r["instance_id"]]
        oracle = max(cache, key=lambda p: score(list(cache[p].values())))
        name = f"hybrid_{family}"
        for tid, m in loro_rows.items():
            per_task[tid].update({f"{name}_loro/{k}": v for k, v in m.items()})
            method_rows[f"{name}_loro"].append(m)
            method_rows_by_task[f"{name}_loro"][tid] = m
        method_rows[f"{name}_insample_best"] = list(cache[oracle].values())
        hybrid_summary[name] = {"loro_params": {repo: {"seeds": p[0], "mix": p[1]} for repo, p in chosen.items()},
                                "insample_best": {"seeds": oracle[0], "mix": oracle[1]},
                                "grid": {f"seeds{p[0]}_mix{p[1]}": round(score(list(m.values())), 3) for p, m in cache.items()}}

    summary_methods = {m: _mean_metrics(rows) for m, rows in method_rows.items()}

    # Paired bootstrap of the LORO hybrid against BM25 (task resampling), unit@k and file@k differences.
    rng = np.random.default_rng(0)
    boot: dict[str, dict] = {}
    tids_all = [r["instance_id"] for r in recs]
    for method in ("hybrid_rep_all_loro", "hybrid_old_callgraph_loro"):
        for metric in ("unit_hit@1", "unit_hit@5", "unit_hit@10", "file_hit@1", "file_hit@5", "file_hit@10"):
            ids = [t for t in tids_all if metric in method_rows_by_task["bm25_units"][t]]
            d = np.array([method_rows_by_task[method][t][metric] - method_rows_by_task["bm25_units"][t][metric] for t in ids])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            boot[f"{method}-bm25/{metric}"] = {"diff": round(float(d.mean()), 3), "ci95": [round(float(np.quantile(bs, 0.025)), 3), round(float(np.quantile(bs, 0.975)), 3)]}

    # Reproduction check against S2 (old call-graph PPR at mix 0.5, 20 seeds).
    repro = None
    if S2_PER_TASK.exists():
        s2 = {json.loads(l)["instance_id"]: json.loads(l) for l in S2_PER_TASK.read_text().splitlines()}
        diffs = 0
        for tid in tids_all:
            for k in (1, 5, 10, 20):
                a = method_rows_by_task["old_callgraph/mix0.5"][tid].get(f"unit_hit@{k}")
                b = s2[tid].get(f"ppr/unit_hit@{k}")
                if a is not None and b is not None and bool(a) != bool(b):
                    diffs += 1
        repro = {"unit_hit_mismatches_vs_s2_ppr": diffs}

    reach: dict[str, dict] = {}
    with_out = [r for r in recs if r["reach"]["gold_outside_top20"] > 0]
    for g in ("old_callgraph", "rep_semantic"):
        for h in (1, 2):
            pairs = [r["reach"][g][h] if h in r["reach"][g] else r["reach"][g][str(h)] for r in with_out]
            reach[f"{g}/{h}hop"] = {
                "mean_new_candidates": round(float(np.mean([p[0] for p in pairs])), 1),
                "tasks_with_gold_reached": sum(p[1] > 0 for p in pairs), "tasks_with_gold_outside_top20": len(with_out)}

    summary = {"methods": summary_methods, "reach_outside_bm25_top20": reach, "hybrid": hybrid_summary, "bootstrap_vs_bm25": boot, "s2_reproduction": repro,
               "mean_unmapped_units": float(np.mean([r["unmapped_units"] for r in recs])),
               "mean_units": float(np.mean([r["n_units"] for r in recs]))}
    return summary, per_task


# ---------------------------------------------------------------------------------------------

def _merge(summary: dict, per_task: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp, tp = OUT / "summary.json", OUT / "per_task.jsonl"
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(summary)
    sp.write_text(json.dumps(old, indent=2))
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in tp.read_text().splitlines()} if tp.exists() else {}
    for tid, d in per_task.items():
        rows.setdefault(tid, {"instance_id": tid}).update(d)
    tp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))


def _print_localize(sm: dict) -> None:
    cols = ["file_hit@1", "file_hit@5", "file_hit@10", "unit_hit@1", "unit_hit@5", "unit_hit@10", "unit_hit@20"]
    print(f"\n{'method':<34}" + "".join(f"{c:>13}" for c in cols))
    for m, v in sm["methods"].items():
        print(f"{m:<34}" + "".join(f"{v.get(c, float('nan')):>13.3f}" for c in cols))


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    _init()
    tids = list(_TASKS)
    t0 = time.perf_counter()
    with Pool(POOL, initializer=_init) as pool:
        if stage in ("build", "all"):
            res = pool.map(_build_one, tids, chunksize=2)
            tot = sum(r[1] for r in res)
            summary = {"build": {"n_graphs_built": len({graph_key(_TASKS[t]) for t in tids}), "cpu_seconds": round(tot, 1),
                                 "mean_seconds_per_task": round(tot / len(res), 2),
                                 "mean_nodes": float(np.mean([r[2] for r in res])), "mean_edges": float(np.mean([r[3] for r in res]))}}
            _merge(summary, {})
            print("build:", summary["build"], f"{time.perf_counter() - t0:.0f}s", flush=True)
        if stage in ("analyze", "all"):
            summary, per_task = analyze(pool, tids)
            _merge({"analyze": summary}, per_task)
            print(json.dumps(summary, indent=2), f"{time.perf_counter() - t0:.0f}s", flush=True)
        if stage in ("localize", "all"):
            summary, per_task = localize(pool, tids)
            _merge({"localize": summary}, per_task)
            _print_localize(summary)
            print(json.dumps({k: summary[k] for k in ("s2_reproduction", "bootstrap_vs_bm25", "mean_unmapped_units")}, indent=1))
            for k, v in summary["hybrid"].items():
                print(k, v["loro_params"], v["insample_best"])
            print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
