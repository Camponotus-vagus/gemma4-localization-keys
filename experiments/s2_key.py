"""S2: identification-key localization with Gemma 4 31B (MLX, local) on top of hybrid retrieval.

Usage:
    uv run python -m experiments.s2_key retrieval            # hybrid rankings for all tasks
    uv run python -m experiments.s2_key run --limit 15       # LLM keys (resumable)
    uv run python -m experiments.s2_key summary
Outputs in results/s2_key/.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from experiments.s2_localize import DATA, ZIP, load_index
from g4a.localize import files_from_units, rank_units_bm25, recall_at_k

OUT = Path("results/s2_key")
# --variant bm25 writes to results/s2_key_bm25 and uses BM25-only candidates (what an in-sandbox skill can compute).
KS = (1, 3, 5, 10)


def gold_for(task: dict, truth: dict, idx) -> tuple[set[str], set[str]]:
    uids = {u.uid for u in idx.units}
    files = {p for p in truth["files"] if p in idx.files}
    units = {f"{s['path']}::{s['qualname']}" for s in truth["symbols"] if s["kind"] != "module"} & uids
    return files, units


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def cmd_retrieval(args) -> None:
    tasks = [json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()]
    if args.variant == "hybrid":
        from g4a.embed_rank import EmbeddingRanker, rrf

        emb = EmbeddingRanker()
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as zf, (OUT / "retrieval.jsonl").open("w") as fh:
        for t in tasks:
            idx = load_index(t, zf)
            bm = rank_units_bm25(idx, t["problem_statement"])[0]
            ranking = rrf(emb.rank_units(idx, t["problem_statement"])[0], bm) if args.variant == "hybrid" else bm
            fh.write(json.dumps({"instance_id": t["instance_id"], "ranking": ranking[:200]}) + "\n")
    print(f"wrote {OUT / 'retrieval.jsonl'}")


def cmd_run(args) -> None:
    from g4a.key_localize import KeyLocalizer

    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    truths = {r["instance_id"]: r for r in load_jsonl(Path("data/derived/groundtruth.jsonl"))}
    retrieval = {r["instance_id"]: r["ranking"] for r in load_jsonl(OUT / "retrieval.jsonl")}
    out_path = OUT / "keys.jsonl"
    done = {(r["instance_id"], r["mode"]) for r in load_jsonl(out_path)}

    # Deterministic task order; only tasks with a gold unit are informative for unit-level keys.
    order = sorted(tasks)
    loc = KeyLocalizer()
    n = 0
    with zipfile.ZipFile(ZIP) as zf, out_path.open("a") as fh:
        for tid in order:
            if args.task_ids and tid not in args.task_ids:
                continue
            t = tasks[tid]
            idx = load_index(t, zf)
            gf, gu = gold_for(t, truths[tid], idx)
            if not gu:
                continue
            n += 1
            if args.limit and n > args.limit:
                break
            for mode in args.modes:
                if (tid, mode) in done:
                    continue
                fn = loc.flat if mode == "flat" else loc.dichotomous
                res = fn(idx, t["problem_statement"], retrieval[tid])
                row = {"instance_id": tid, "mode": mode, "ranking": res.ranking[:50], "picks": res.picks,
                       "calls": res.calls, "prompt_tokens": res.prompt_tokens,
                       "completion_tokens": res.completion_tokens, "seconds": round(res.seconds, 1),
                       "raw": res.raw}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                hit1 = bool(set(res.ranking[:1]) & gu)
                print(f"{tid} {mode}: picks={len(res.picks)} unit_hit@1={hit1} "
                      f"{res.prompt_tokens} tok {res.seconds:.0f}s", flush=True)


def cmd_summary(_args) -> None:
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    truths = {r["instance_id"]: r for r in load_jsonl(Path("data/derived/groundtruth.jsonl"))}
    retrieval = {r["instance_id"]: r["ranking"] for r in load_jsonl(OUT / "retrieval.jsonl")}
    keys = load_jsonl(OUT / "keys.jsonl")
    modes = sorted({r["mode"] for r in keys})
    # Evaluate only tasks where every mode has a result, so the comparison is paired.
    by = {(r["instance_id"], r["mode"]): r for r in keys}
    common = sorted(t for t in {r["instance_id"] for r in keys} if all((t, m) in by for m in modes))
    rows = {m: [] for m in ["retrieval", *modes]}
    with zipfile.ZipFile(ZIP) as zf:
        for tid in common:
            idx = load_index(tasks[tid], zf)
            gf, gu = gold_for(tasks[tid], truths[tid], idx)
            for m in rows:
                rk = retrieval[tid] if m == "retrieval" else by[(tid, m)]["ranking"]
                files = files_from_units(rk)
                rows[m].append({**{f"unit@{k}": recall_at_k(rk, gu, k)[1] for k in KS},
                                **{f"file@{k}": recall_at_k(files, gf, k)[1] for k in KS}})
    print(f"paired tasks: {len(common)}")
    cols = [f"file@{k}" for k in (1, 3)] + [f"unit@{k}" for k in KS]
    print(f"{'mode':<13}" + "".join(f"{c:>9}" for c in cols) + f"{'calls':>7}{'ptok':>7}{'sec':>6}")
    summary = {}
    for m, rs in rows.items():
        vals = {c: float(np.mean([r[c] for r in rs])) for c in cols}
        cost = [by[(t, m)] for t in common] if m != "retrieval" else []
        c_calls = np.mean([r["calls"] for r in cost]) if cost else 0
        c_tok = np.mean([r["prompt_tokens"] for r in cost]) if cost else 0
        c_sec = np.mean([r["seconds"] for r in cost]) if cost else 0
        summary[m] = {**vals, "calls": c_calls, "prompt_tokens": c_tok, "seconds": c_sec, "n": len(rs)}
        print(f"{m:<13}" + "".join(f"{vals[c]:>9.3f}" for c in cols) + f"{c_calls:>7.1f}{c_tok:>7.0f}{c_sec:>6.0f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("retrieval")
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--task-ids", nargs="+")
    r.add_argument("--modes", nargs="+", default=["flat", "dichotomous"])
    sub.add_parser("summary")
    ap.add_argument("--variant", choices=["hybrid", "bm25"], default="hybrid")
    args = ap.parse_args()
    global OUT
    if args.variant == "bm25":
        OUT = Path("results/s2_key_bm25")
    {"retrieval": cmd_retrieval, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
