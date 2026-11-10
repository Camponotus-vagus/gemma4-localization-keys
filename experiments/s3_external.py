"""S3b: reproduce the S2 localization benchmark on public repos outside the competition (SWE-bench Verified).

Usage (uv run python -m experiments.s3_external <cmd>):
    select                      sample task ids -> experiments/s3_external_tasks.json (needs data/external/swebench_verified.jsonl)
    snapshots                   git archive of *.py at each base_commit -> data/external/snapshots/
    truth                       gold-patch ground truth -> data/external/groundtruth.jsonl, prints the kind table
    baselines                   BM25 files/units, PPR, embedding, RRF -> results/s3_external/{per_task.jsonl,summary.json}
    retrieval                   top-200 unit rankings for the key runs (hybrid and bm25) -> results/s3_external/retrieval_*.jsonl
    key [--limit N] [--task-ids ...] [--variants ...] [--modes ...]    LLM keys (needs mlx server-free Gemma; resumable)
    summary                     table + paired bootstrap for the key runs
Query text is the problem statement only (no hints_text), as in S2.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from g4a import external as ext
from g4a.localize import files_from_units, rank_files_bm25, rank_units_bm25, rank_units_ppr, recall_at_k
from g4a.repo_index import RepoIndex, cached_index

OUT = Path("results/s3_external")
INDEX_DIR = ext.EXT / "index"
EMB_CACHE = ext.EXT / "emb_cache"  # separate from the competition cache
KS = (1, 5, 10, 20)
KEY_KS = (1, 3, 5, 10)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def load_index(task: dict) -> RepoIndex:
    key = f"{task['repo'].split('/')[1]}_{task['base_commit']}"
    return cached_index(ext.snapshot_path(task), INDEX_DIR, key)


def gold_for(truth: dict, idx: RepoIndex) -> tuple[set[str], set[str]]:
    uids = {u.uid for u in idx.units}
    files = {p for p in truth["files"] if p in idx.files}
    units = {f"{s['path']}::{s['qualname']}" for s in truth["symbols"] if s["kind"] != "module"} & uids
    return files, units


def embedder():
    import g4a.embed_rank as er

    er.CACHE_DIR = EMB_CACHE
    ranker = er.EmbeddingRanker()
    ranker.model.half()  # fp16 on MPS is ~2x faster than fp32; cosine to the fp32 embeddings is >0.9999
    return ranker


# ---------------------------------------------------------------- select / snapshots / truth

def cmd_select(_a) -> None:
    ids = ext.select_ids()
    data = ext.load_dataset()
    counts: dict[str, int] = defaultdict(int)
    for i in ids:
        counts[data[i]["repo"]] += 1
    ext.TASKS_JSON.write_text(json.dumps({
        "source": "princeton-nlp/SWE-bench_Verified (test)", "seed": ext.SEED,
        "selection": {k: ("all" if v is None else v) for k, v in ext.SELECTION.items()},
        "counts": dict(counts), "instance_ids": ids}, indent=1) + "\n")
    print(dict(counts), len(ids))


def cmd_snapshots(_a) -> None:
    tasks = ext.load_tasks()
    t0 = time.perf_counter()
    for n, t in enumerate(tasks, 1):
        had = ext.snapshot_path(t).exists()
        p = ext.ensure_snapshot(t)
        print(f"[{n}/{len(tasks)}] {t['instance_id']} {'cached' if had else 'built'} {p.stat().st_size / 1e6:.1f} MB "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


def cmd_truth(_a) -> None:
    tasks = ext.load_tasks()
    rows = [json.loads(json.dumps(__import__("dataclasses").asdict(ext.build_truth_external(t)))) for t in tasks]
    ext.TRUTH_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = ext.symbol_kind_counts(rows)
    print("gold symbols by kind:", dict(c), "total", sum(c.values()))
    n_unit = n_file = 0
    per_repo: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for t, r in zip(tasks, rows):
        idx = load_index(t)
        gf, gu = gold_for(r, idx)
        n_file += bool(gf)
        n_unit += bool(gu)
        pr = per_repo[t["repo"]]
        pr[0] += 1
        pr[1] += bool(gf)
        pr[2] += bool(gu)
    print(f"tasks {len(rows)}; with gold file in index {n_file}; with gold unit {n_unit}")
    for k, v in per_repo.items():
        print(f"  {k}: tasks {v[0]}, gold file {v[1]}, gold unit {v[2]}")


# ---------------------------------------------------------------- baselines

def cmd_baselines(args) -> None:
    from g4a.embed_rank import rrf

    methods = args.methods
    emb = embedder() if any(m.startswith("emb") for m in methods) else None
    tasks = ext.load_tasks()
    truths = {r["instance_id"]: r for r in load_jsonl(ext.TRUTH_PATH)}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for n, t in enumerate(tasks, 1):
        idx = load_index(t)
        gf, gu = gold_for(truths[t["instance_id"]], idx)
        issue = t["problem_statement"]
        row = {"instance_id": t["instance_id"], "repo": t["repo"], "n_gold_files": len(gf),
               "n_gold_units": len(gu), "n_units": len(idx.units), "n_files": len(idx.files)}
        bm_units = None
        for m in methods:
            if m == "bm25_files":
                files, units = rank_files_bm25(idx, issue), None
            elif m == "bm25_units":
                units = bm_units = rank_units_bm25(idx, issue)[0]
                files = files_from_units(units)
            elif m == "ppr":
                units = rank_units_ppr(idx, issue)
                files = files_from_units(units)
            elif m == "emb":
                units = emb.rank_units(idx, issue)[0]
                files = files_from_units(units)
            elif m == "emb+bm25":
                units = rrf(emb.rank_units(idx, issue)[0], bm_units or rank_units_bm25(idx, issue)[0])
                files = files_from_units(units)
            else:
                raise SystemExit(f"unknown method {m}")
            for k in KS:
                r, h = recall_at_k(files, gf, k)
                row[f"{m}/file_recall@{k}"], row[f"{m}/file_hit@{k}"] = r, h
                if units is not None:
                    r, h = recall_at_k(units, gu, k)
                    row[f"{m}/unit_recall@{k}"], row[f"{m}/unit_hit@{k}"] = r, h
        rows.append(row)
        print(f"[{n}/{len(tasks)}] {t['instance_id']} {time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = summarize_baselines(rows, methods)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print_baselines(summ, methods)


def summarize_baselines(rows: list[dict], methods: list[str], repo: str | None = None) -> dict:
    rows = [r for r in rows if repo is None or r["repo"] == repo]
    out: dict[str, dict] = defaultdict(dict)
    for key in sorted({k for r in rows for k in r if "/" in k}):
        vals = [r[key] for r in rows if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))
                and (r["n_gold_units"] if "unit_" in key else r["n_gold_files"])]
        m, metric = key.split("/")
        out[m][metric] = round(float(np.mean(vals)), 3) if vals else float("nan")
        out[m]["n_units" if "unit_" in key else "n_files"] = len(vals)
    return dict(out)


def print_baselines(summ: dict, methods: list[str]) -> None:
    cols = ["file_hit@1", "file_hit@5", "file_hit@10", "unit_hit@1", "unit_hit@5", "unit_hit@10", "unit_hit@20", "unit_recall@10"]
    print(f"\n{'method':<14}" + "".join(f"{c:>15}" for c in cols))
    for m in methods:
        print(f"{m:<14}" + "".join(f"{summ[m].get(c, float('nan')):>15.3f}" for c in cols))
    m0 = summ[methods[0]]
    print(f"n with gold file: {m0.get('n_files')}, with gold unit: {summ.get('bm25_units', {}).get('n_units')}")


def cmd_baseline_summary(_a) -> None:
    rows = load_jsonl(OUT / "per_task.jsonl")
    methods = [m for m in ["bm25_files", "bm25_units", "ppr", "emb", "emb+bm25"] if any(k.startswith(m + "/") for k in rows[0])]
    print_baselines(summarize_baselines(rows, methods), methods)
    for repo in sorted({r["repo"] for r in rows}):
        s = summarize_baselines(rows, methods, repo)
        print(f"\n{repo} (tasks {sum(r['repo'] == repo for r in rows)})")
        print_baselines(s, methods)


# ---------------------------------------------------------------- keys

VARIANTS = ("hybrid", "bm25")


def cmd_retrieval(args) -> None:
    from g4a.embed_rank import rrf

    emb = embedder() if "hybrid" in args.variants else None
    tasks = [t for t in ext.load_tasks() if not args.task_ids or t["instance_id"] in args.task_ids]
    OUT.mkdir(parents=True, exist_ok=True)
    done = {v: {r["instance_id"] for r in load_jsonl(OUT / f"retrieval_{v}.jsonl")} for v in args.variants}
    fhs = {v: (OUT / f"retrieval_{v}.jsonl").open("a") for v in args.variants}  # append: resumable
    for n, t in enumerate(tasks, 1):
        idx = load_index(t)
        bm = rank_units_bm25(idx, t["problem_statement"])[0]
        for v, fh in fhs.items():
            if t["instance_id"] in done[v]:
                continue
            rk = bm if v == "bm25" else rrf(emb.rank_units(idx, t["problem_statement"])[0], bm)
            fh.write(json.dumps({"instance_id": t["instance_id"], "ranking": rk[:200]}) + "\n")
            fh.flush()
        print(f"[{n}/{len(tasks)}] {t['instance_id']}", flush=True)
    for fh in fhs.values():
        fh.close()


def cmd_key(args) -> None:
    from g4a.key_localize import KeyLocalizer

    tasks = {t["instance_id"]: t for t in ext.load_tasks()}
    truths = {r["instance_id"]: r for r in load_jsonl(ext.TRUTH_PATH)}
    retrieval = {v: {r["instance_id"]: r["ranking"] for r in load_jsonl(OUT / f"retrieval_{v}.jsonl")} for v in args.variants}
    out_path = OUT / "keys.jsonl"
    done = {(r["instance_id"], r["variant"], r["mode"]) for r in load_jsonl(out_path)}
    loc = KeyLocalizer()
    n = 0
    with out_path.open("a") as fh:
        for tid in sorted(tasks):  # deterministic order; only tasks with a gold unit are informative
            if args.task_ids and tid not in args.task_ids:
                continue
            t = tasks[tid]
            idx = load_index(t)
            if not gold_for(truths[tid], idx)[1]:
                continue
            n += 1
            if args.limit and n > args.limit:
                break
            for variant in args.variants:
                for mode in args.modes:
                    if (tid, variant, mode) in done:
                        continue
                    fn = loc.flat if mode == "flat" else loc.dichotomous
                    res = fn(idx, t["problem_statement"], retrieval[variant][tid])
                    fh.write(json.dumps({"instance_id": tid, "variant": variant, "mode": mode,
                                         "ranking": res.ranking[:50], "picks": res.picks, "calls": res.calls,
                                         "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                                         "seconds": round(res.seconds, 1), "raw": res.raw}) + "\n")
                    fh.flush()
                    gu = gold_for(truths[tid], idx)[1]
                    print(f"{tid} {variant}/{mode}: picks={len(res.picks)} unit_hit@1={bool(set(res.ranking[:1]) & gu)} "
                          f"{res.prompt_tokens} tok {res.seconds:.0f}s", flush=True)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cmd_summary(_a) -> None:
    tasks = {t["instance_id"]: t for t in ext.load_tasks()}
    truths = {r["instance_id"]: r for r in load_jsonl(ext.TRUTH_PATH)}
    keys = load_jsonl(OUT / "keys.jsonl")
    by = {(r["instance_id"], r["variant"], r["mode"]): r for r in keys}
    conds = sorted({(r["variant"], r["mode"]) for r in keys})
    common = sorted(t for t in {r["instance_id"] for r in keys} if all((t, v, m) in by for v, m in conds))
    retr = {v: {r["instance_id"]: r["ranking"] for r in load_jsonl(OUT / f"retrieval_{v}.jsonl")} for v in {v for v, _ in conds}}
    names = [f"retrieval[{v}]" for v in sorted(retr)] + [f"{v}/{m}" for v, m in conds]
    rows: dict[str, list[dict]] = {k: [] for k in names}
    for tid in common:
        idx = load_index(tasks[tid])
        gf, gu = gold_for(truths[tid], idx)
        for name in names:
            if name.startswith("retrieval"):
                rk = retr[name[10:-1]][tid]
            else:
                v, m = name.split("/")
                rk = by[(tid, v, m)]["ranking"]
            files = files_from_units(rk)
            rows[name].append({**{f"unit@{k}": recall_at_k(rk, gu, k)[1] for k in KEY_KS},
                               **{f"file@{k}": recall_at_k(files, gf, k)[1] for k in (1, 3)}})
    print(f"paired tasks: {len(common)}")
    cols = ["file@1", "file@3"] + [f"unit@{k}" for k in KEY_KS]
    print(f"{'condition':<20}" + "".join(f"{c:>9}" for c in cols) + f"{'calls':>7}{'ptok':>7}{'sec':>6}")
    summary = {}
    for name, rs in rows.items():
        vals = {c: float(np.mean([r[c] for r in rs])) for c in cols}
        cost = [] if name.startswith("retrieval") else [by[(t, *name.split("/"))] for t in common]
        calls, tok, sec = (float(np.mean([r[x] for r in cost])) if cost else 0.0 for x in ("calls", "prompt_tokens", "seconds"))
        summary[name] = {**vals, "calls": calls, "prompt_tokens": tok, "seconds": sec, "n": len(rs)}
        print(f"{name:<20}" + "".join(f"{vals[c]:>9.3f}" for c in cols) + f"{calls:>7.1f}{tok:>7.0f}{sec:>6.0f}")
    print("\nPaired bootstrap (5000 resamples), unit@1")
    for v in sorted(retr):
        a, b, r = (np.array([x["unit@1"] for x in rows[n]]) for n in (f"{v}/flat", f"{v}/dichotomous", f"retrieval[{v}]"))
        if len(a) and len(b):
            for label, x, y in ((f"{v}: flat - dichotomous", a, b), (f"{v}: flat - retrieval", a, r), (f"{v}: dichotomous - retrieval", b, r)):
                d, lo, hi = paired_bootstrap(x, y)
                print(f"  {label}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; discordant {int(((x > y)).sum())} vs {int((y > x).sum())}")
    (OUT / "key_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("select", "snapshots", "truth", "summary"):
        sub.add_parser(c)
    sub.add_parser("baseline-summary")
    b = sub.add_parser("baselines")
    b.add_argument("--methods", nargs="+", default=["bm25_files", "bm25_units", "ppr", "emb", "emb+bm25"])
    r = sub.add_parser("retrieval")
    r.add_argument("--variants", nargs="+", default=list(VARIANTS))
    r.add_argument("--task-ids", nargs="+")
    k = sub.add_parser("key")
    k.add_argument("--limit", type=int, default=0)
    k.add_argument("--task-ids", nargs="+")
    k.add_argument("--variants", nargs="+", default=list(VARIANTS))
    k.add_argument("--modes", nargs="+", default=["flat", "dichotomous"])
    args = ap.parse_args()
    {"select": cmd_select, "snapshots": cmd_snapshots, "truth": cmd_truth, "baselines": cmd_baselines,
     "baseline-summary": cmd_baseline_summary, "retrieval": cmd_retrieval, "key": cmd_key,
     "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
