"""S3c: graph-expanded candidate sets for the identification key (flat / multi-access), BM25 only.

Usage (after `source scripts/env.sh`):
    uv run python -m experiments.s3_graphkey retrieval                  # candidate sets + ceilings, no LLM
    uv run python -m experiments.s3_graphkey run --variants NAME ...    # LLM flat key (resumable; needs MLX)
    uv run python -m experiments.s3_graphkey summary
    add `--set external` before the subcommand for the SWE-bench Verified tasks (experiments/s3_external_tasks.json).
Outputs: results/s3_graphkey/ (train) and results/s3_graphkey/external/.
The graph is used to expand the candidates the LLM chooses from, not as a score (see paper/notes/s3_graph_key.md).
"""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g4a.graph_key import GraphView, build_cands, flat_key, prompt_chars, variant_specs
from g4a.localize import files_from_units, rank_units_bm25, recall_at_k
from g4a.repo_index import RepoIndex

OUT = Path("results/s3_graphkey")
S2_BM25_KEYS = Path("results/s2_key_bm25/keys.jsonl")
KS = (1, 3, 5, 10)


@dataclass
class Item:
    tid: str
    task: dict
    idx: RepoIndex
    graph: dict
    gold_files: set[str]
    gold_units: set[str]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _gold(truth: dict, idx: RepoIndex) -> tuple[set[str], set[str]]:
    uids = {u.uid for u in idx.units}
    files = {p for p in truth["files"] if p in idx.files}
    units = {f"{s['path']}::{s['qualname']}" for s in truth["symbols"] if s["kind"] != "module"} & uids
    return files, units


def iter_items(which: str, only: set[str] | None = None):
    """Yield one Item per task (sorted by id), loading index, repaired graph and gold lazily."""
    if which == "train":
        from experiments.s2_localize import DATA, ZIP, load_index
        from experiments.s3_graph import repaired_graph

        tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
        truths = {r["instance_id"]: r for r in load_jsonl(Path("data/derived/groundtruth.jsonl"))}
        with zipfile.ZipFile(ZIP) as zf:
            for tid in sorted(tasks):
                if only and tid not in only:
                    continue
                t = tasks[tid]
                idx = load_index(t, zf)
                gf, gu = _gold(truths[tid], idx)
                if gu:  # only tasks with a gold unit are informative (as S2)
                    yield Item(tid, t, idx, repaired_graph(t), gf, gu)
    else:
        from g4a import external as ext
        from g4a.graph_builder import build_graph_from_snapshot
        from g4a.repo_index import cached_index

        truths = {r["instance_id"]: r for r in load_jsonl(ext.TRUTH_PATH)}
        gdir = ext.EXT / "graphs_repaired"
        gdir.mkdir(parents=True, exist_ok=True)
        tasks = {t["instance_id"]: t for t in ext.load_tasks()}
        for tid in sorted(tasks):
            if only and tid not in only:
                continue
            t = tasks[tid]
            key = f"{t['repo'].split('/')[1]}_{t['base_commit']}"
            idx = cached_index(ext.snapshot_path(t), ext.EXT / "index", key)
            gf, gu = _gold(truths[tid], idx)
            if not gu:
                continue
            cache = gdir / f"{key}.json.gz"
            if cache.exists():
                g = json.loads(gzip.decompress(cache.read_bytes()))
            else:
                g = build_graph_from_snapshot(ext.snapshot_path(t), key)
                cache.write_bytes(gzip.compress(json.dumps(g).encode()))
            yield Item(tid, t, idx, g, gf, gu)


def _metrics(ranking: list[str], it: Item) -> dict[str, float]:
    files = files_from_units(ranking)
    return {**{f"unit@{k}": float(recall_at_k(ranking, it.gold_units, k)[1]) for k in KS},
            **{f"file@{k}": float(recall_at_k(files, it.gold_files, k)[1]) for k in (1, 3)}}


def cmd_retrieval(args) -> None:
    specs = variant_specs()
    names = args.variants or list(specs)
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list[dict]] = {n: [] for n in names}
    # Partial runs (--variants) replace only those variants and keep the other stored rows.
    prev = [r for r in load_jsonl(out / "retrieval.jsonl") if r["variant"] not in names]
    with (out / "retrieval.jsonl").open("w") as fh:
        for r in prev:
            fh.write(json.dumps(r) + "\n")
        for it in iter_items(args.set):
            ranking, bm = rank_units_bm25(it.idx, it.task["problem_statement"])
            uids = [u.uid for u in it.idx.units]
            view = GraphView(it.idx, it.graph)
            for n in names:
                c = build_cands(specs[n], view, bm, uids)
                k = specs[n]["k"]
                cands = c.display
                files = files_from_units(cands)
                row = {"instance_id": it.tid, "variant": n, "ranking": c.priority[:200], "display": cands,
                       "attach": c.attach, "n_cands": len(cands),
                       "cand_hit": bool(set(cands) & it.gold_units),
                       "cand_file_hit": bool(set(files) & it.gold_files) if it.gold_files else None,
                       "gold_in_bm25_topk": bool(set(ranking[:k]) & it.gold_units),
                       "chars": prompt_chars(it.idx, c, specs[n]["kind"] == "grouped")}
                fh.write(json.dumps(row) + "\n")
                acc[n].append({**row, **_metrics(c.priority, it)})
    print(f"wrote {out / 'retrieval.jsonl'}")
    summ = {}
    cols = ["unit@1", "unit@5", "unit@10"]
    print(f"{'variant':<28}{'n':>4}{'cand_unit':>10}{'cand_file':>10}{'chars':>7}" + "".join(f"{c:>9}" for c in cols))
    for n, rs in acc.items():
        fr = [r["cand_file_hit"] for r in rs if r["cand_file_hit"] is not None]
        s = {"n": len(rs), "cand_unit_ceiling": float(np.mean([r["cand_hit"] for r in rs])),
             "cand_unit_count": int(sum(r["cand_hit"] for r in rs)),
             "cand_file_ceiling": float(np.mean(fr)), "n_file": len(fr), "mean_chars": float(np.mean([r["chars"] for r in rs])),
             **{c: float(np.mean([r[c] for r in rs])) for c in cols}}
        summ[n] = s
        print(f"{n:<28}{s['n']:>4}{s['cand_unit_ceiling']:>10.3f}{s['cand_file_ceiling']:>10.3f}{s['mean_chars']:>7.0f}"
              + "".join(f"{s[c]:>9.3f}" for c in cols))
    # Paired candidate-set comparison against bm25 with the same card count K and with the closest mean prompt chars.
    bases = {n: {r["instance_id"]: r["cand_hit"] for r in rs} for n, rs in acc.items() if n.startswith("bm25_k")}
    print(f"\n{'variant':<28}{'vs same K':>16}{'vs same chars':>26}   (candidate-set hit: gained/lost tasks)")
    for n, rs in acc.items():
        if n.startswith("bm25_k") or not bases:
            continue
        def gl(b: str) -> tuple[int, int]:
            return (sum(r["cand_hit"] and not bases[b][r["instance_id"]] for r in rs),
                    sum(bases[b][r["instance_id"]] and not r["cand_hit"] for r in rs))
        sameK = f"bm25_k{specs[n]['k']}"
        near = min(bases, key=lambda b: abs(summ[b]["mean_chars"] - summ[n]["mean_chars"]))
        out_s = {}
        txt = ""
        if sameK in bases:
            out_s["vs_same_k"] = {"baseline": sameK, "gained": gl(sameK)[0], "lost": gl(sameK)[1]}
            txt += f"{sameK}: +{gl(sameK)[0]}/-{gl(sameK)[1]}"
        out_s["vs_same_chars"] = {"baseline": near, "gained": gl(near)[0], "lost": gl(near)[1]}
        summ[n].update(out_s)
        print(f"{n:<28}{txt:>16}{near + f': +{gl(near)[0]}/-{gl(near)[1]}':>26}")
    sp = out / "retrieval_summary.json"
    old_summ = json.loads(sp.read_text()) if sp.exists() else {}
    sp.write_text(json.dumps({**{k: v for k, v in old_summ.items() if k not in summ}, **summ}, indent=2))


def cmd_run(args) -> None:
    from g4a.key_localize import KeyLocalizer

    specs = variant_specs()
    names = args.variants
    for n in names:
        if n not in specs:
            raise SystemExit(f"unknown variant {n}")
    out_path = OUT / "keys.jsonl"
    cands = {(r["instance_id"], r["variant"]): r for r in load_jsonl(OUT / "retrieval.jsonl")}
    done = {(r["instance_id"], r["mode"]) for r in load_jsonl(out_path)}
    # The bm25_k20 prompt is identical to the S2 / S3b bm25 flat prompt (greedy decoding): reuse those rows when present.
    if args.reuse and "bm25_k20" in names:
        src = S2_BM25_KEYS if args.set == "train" else Path("results/s3_external/keys.jsonl")
        with out_path.open("a") as fh:
            for r in load_jsonl(src):
                if args.set == "external" and r.get("variant") != "bm25":
                    continue
                if r["mode"] == "flat" and (r["instance_id"], "bm25_k20") not in done:
                    fh.write(json.dumps({**r, "mode": "bm25_k20", "reused_from": str(src)}) + "\n")
                    done.add((r["instance_id"], "bm25_k20"))
        names = [n for n in names if n != "bm25_k20"]
    loc = None
    n_task = 0
    with out_path.open("a") as fh:
        for it in iter_items(args.set, set(args.task_ids) if args.task_ids else None):
            n_task += 1
            if args.limit and n_task > args.limit:
                break
            for name in names:
                if (it.tid, name) in done:
                    continue
                row = cands.get((it.tid, name))
                if row is None:
                    raise SystemExit(f"no candidates for {it.tid}/{name}: run `retrieval` first")
                from g4a.graph_key import Cands

                c = Cands(priority=row["ranking"], display=row["display"], attach={k: tuple(v) for k, v in row["attach"].items()})
                # ranking in retrieval.jsonl is truncated to 200: the tail beyond is irrelevant for unit@<=10
                loc = loc or KeyLocalizer()
                res = flat_key(loc, it.idx, it.task["problem_statement"], c, specs[name]["kind"] == "grouped")
                fh.write(json.dumps({"instance_id": it.tid, "mode": name, "ranking": res.ranking[:50], "picks": res.picks,
                                     "calls": res.calls, "prompt_tokens": res.prompt_tokens,
                                     "completion_tokens": res.completion_tokens, "seconds": round(res.seconds, 1),
                                     "raw": res.raw}) + "\n")
                fh.flush()
                print(f"{it.tid} {name}: picks={len(res.picks)} unit_hit@1={bool(set(res.ranking[:1]) & it.gold_units)} "
                      f"{res.prompt_tokens} tok {res.seconds:.0f}s", flush=True)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cmd_summary(args) -> None:
    keys = load_jsonl(OUT / "keys.jsonl")
    by = {(r["instance_id"], r["mode"]): r for r in keys}
    modes = sorted({r["mode"] for r in keys})
    common = sorted(t for t in {r["instance_id"] for r in keys} if all((t, m) in by for m in modes))
    ret = {(r["instance_id"], r["variant"]): r for r in load_jsonl(OUT / "retrieval.jsonl")}
    items = {it.tid: it for it in iter_items(args.set, set(common))}
    common = [t for t in common if t in items]
    rows: dict[str, list[dict]] = {}
    for m in modes:
        rows[m] = [_metrics(by[(t, m)]["ranking"], items[t]) for t in common]
        rows[f"retrieval[{m}]"] = [_metrics(ret[(t, m)]["ranking"], items[t]) for t in common]
    print(f"paired tasks: {len(common)}")
    cols = ["file@1", "file@3"] + [f"unit@{k}" for k in KS]
    print(f"{'condition':<30}" + "".join(f"{c:>9}" for c in cols) + f"{'ceil':>7}{'calls':>6}{'ptok':>7}{'sec':>6}")
    summary = {}
    for name, rs in rows.items():
        vals = {c: float(np.mean([r[c] for r in rs])) for c in cols}
        cost = [] if name.startswith("retrieval") else [by[(t, name)] for t in common]
        ceil = float(np.mean([ret[(t, name.removeprefix("retrieval[").removesuffix("]"))]["cand_hit"] for t in common]))
        calls, tok, sec = (float(np.mean([r[x] for r in cost])) if cost else 0.0 for x in ("calls", "prompt_tokens", "seconds"))
        summary[name] = {**vals, "ceiling": ceil, "calls": calls, "prompt_tokens": tok, "seconds": sec, "n": len(rs)}
        print(f"{name:<30}" + "".join(f"{vals[c]:>9.3f}" for c in cols) + f"{ceil:>7.3f}{calls:>6.1f}{tok:>7.0f}{sec:>6.0f}")
    print("\nPaired bootstrap (5000 resamples), unit@1; baselines = bm25 flat with the same card count and bm25_k20")
    for m in modes:
        if m.startswith("bm25_k"):
            continue
        same_k = f"bm25_k{m.rsplit('_k', 1)[1].split('_')[0]}"
        for base in dict.fromkeys([same_k, "bm25_k20"]):  # same card count, and the cheapest (S2-equivalent) baseline
            if base not in rows:
                continue
            a, b = np.array([r["unit@1"] for r in rows[m]]), np.array([r["unit@1"] for r in rows[base]])
            d, lo, hi = paired_bootstrap(a, b)
            print(f"  {m} - {base}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; discordant {int((a > b).sum())} vs {int((b > a).sum())}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["train", "external"], default="train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--variants", nargs="+")
    u = sub.add_parser("run")
    u.add_argument("--variants", nargs="+", required=True)
    u.add_argument("--limit", type=int, default=0)
    u.add_argument("--task-ids", nargs="+")
    u.add_argument("--no-reuse", dest="reuse", action="store_false", help="rerun bm25_k20 instead of reusing S2 / S3b rows")
    sub.add_parser("summary")
    args = ap.parse_args()
    if args.set == "external":
        OUT = OUT / "external"
    {"retrieval": cmd_retrieval, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
