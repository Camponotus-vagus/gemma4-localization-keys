"""S5: paired bootstrap CIs and discordant-pair counts for the comparisons in Sections 4.2, 4.3 and 5.

Usage: source scripts/env.sh && uv run --no-sync python -m experiments.s5_stats
Reads saved per-task hits (results/s2_localize, results/s3_external, results/s3_graph) and saved key
results (results/s2_key*, results/s3_external/keys.jsonl); nothing is recomputed with models.
Writes results/s5_stats/summary.json and paper/notes/s5_stats.md. 5,000 resamples, seed 0.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import s2_key, s3_external as s3
from experiments.s2_localize import DATA, ZIP, load_index as train_index
from g4a import external as ext
from g4a.localize import files_from_units, recall_at_k

B, SEED = 5000, 0
OUT = Path("results/s5_stats")
NOTE = Path("paper/notes/s5_stats.md")
METRICS = ("unit@1", "unit@10", "file@1", "file@10")
# (first, second) -> reported as first minus second
RETRIEVAL_PAIRS = [("emb+bm25", "bm25_units"), ("emb", "bm25_units"), ("ppr", "bm25_units"),
                   ("emb+bm25", "emb"), ("provided_tool", "bm25_units")]
LABEL = {"emb+bm25": "RRF", "emb": "Qwen3", "ppr": "PPR", "bm25_units": "BM25", "provided_tool": "provided"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def boot(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return {"diff": round(float(d.mean()), 4), "lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4), "n": int(len(d)),
            "first_only": int((a > b).sum()), "second_only": int((b > a).sum())}


def retrieval_block(rows: list[dict], pairs) -> dict:
    out = {}
    for x, y in pairs:
        for met in METRICS:
            kind, k = met.split("@")
            hk = f"{kind}_hit@{k}"
            gate = "n_gold_units" if kind == "unit" else "n_gold_files"
            use = [r for r in rows if r[gate] and f"{x}/{hk}" in r and f"{y}/{hk}" in r]
            if not use:
                continue
            a = np.array([bool(r[f"{x}/{hk}"]) for r in use])
            b = np.array([bool(r[f"{y}/{hk}"]) for r in use])
            out[f"{x} - {y} | {met}"] = boot(a, b)
    return out


def key_hits(rows: list[dict], retr: dict, gold: dict) -> dict[str, dict[str, np.ndarray]]:
    """-> {condition: {metric: hits over sorted common tids}}; conditions: retrieval, flat, dichotomous."""
    by = {(r["instance_id"], r["mode"]): r for r in rows}
    tids = sorted(t for t in retr if (t, "flat") in by and (t, "dichotomous") in by)
    res: dict[str, dict[str, list]] = {c: defaultdict(list) for c in ("retrieval", "flat", "dichotomous")}
    for t in tids:
        gf, gu = gold[t]
        for c, rk in (("retrieval", retr[t]), ("flat", by[(t, "flat")]["ranking"]), ("dichotomous", by[(t, "dichotomous")]["ranking"])):
            files = files_from_units(rk)
            res[c]["unit@1"].append(recall_at_k(rk, gu, 1)[1])
            res[c]["unit@10"].append(recall_at_k(rk, gu, 10)[1])
            res[c]["file@1"].append(recall_at_k(files, gf, 1)[1])
            res[c]["file@10"].append(recall_at_k(files, gf, 10)[1])
    res["_tids"] = tids
    return {c: ({m: np.array(v) for m, v in d.items()} if c != "_tids" else d) for c, d in res.items()}


def key_block(h: dict) -> dict:
    out = {}
    for x, y in (("flat", "dichotomous"), ("flat", "retrieval"), ("dichotomous", "retrieval")):
        for met in ("unit@1", "unit@10", "file@1"):
            out[f"{x} - {y} | {met}"] = boot(h[x][met], h[y][met])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"resamples": B, "seed": SEED}

    # ---- 4.2 retrieval baselines (saved per-task hits)
    train_rows = jl(Path("results/s2_localize/per_task.jsonl"))
    ext_rows = jl(Path("results/s3_external/per_task.jsonl"))
    summary["retrieval"] = {"train": retrieval_block(train_rows, RETRIEVAL_PAIRS),
                            "external": retrieval_block(ext_rows, RETRIEVAL_PAIRS[:4])}

    # ---- 4.3 repaired graph, LORO hybrid vs BM25 (train only), same 5,000-resample procedure
    g_rows = jl(Path("results/s3_graph/per_task.jsonl"))
    gb = {}
    for h in ("hybrid_rep_all_loro", "hybrid_old_callgraph_loro"):
        for met in METRICS:
            kind, k = met.split("@")
            hk = f"{kind}_hit@{k}"
            use = [r for r in g_rows if f"{h}/{hk}" in r and f"bm25_units/{hk}" in r]
            gb[f"{h} - bm25_units | {met}"] = boot(np.array([r[f"{h}/{hk}"] for r in use]) > 0.5,
                                                   np.array([r[f"bm25_units/{hk}"] for r in use]) > 0.5)
    summary["graph_loro"] = gb

    # ---- 5 keys
    tasks_t = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    truths_t = {r["instance_id"]: r for r in jl(Path("data/derived/groundtruth.jsonl"))}
    keys: dict = {"train": {}, "external": {}}
    hits: dict = {"train": {}, "external": {}}
    with zipfile.ZipFile(ZIP) as zf:
        for v, d in (("hybrid", Path("results/s2_key")), ("bm25", Path("results/s2_key_bm25"))):
            rows = jl(d / "keys.jsonl")
            retr = {r["instance_id"]: r["ranking"] for r in jl(d / "retrieval.jsonl")}
            ids = {r["instance_id"] for r in rows}
            gold = {t: s2_key.gold_for(tasks_t[t], truths_t[t], train_index(tasks_t[t], zf)) for t in ids}
            hits["train"][v] = key_hits(rows, {t: retr[t] for t in ids}, gold)
    tasks_e = {t["instance_id"]: t for t in ext.load_tasks()}
    truths_e = {r["instance_id"]: r for r in jl(ext.TRUTH_PATH)}
    krows = jl(s3.OUT / "keys.jsonl")
    for v in ("hybrid", "bm25"):
        rows = [r for r in krows if r["variant"] == v]
        retr = {r["instance_id"]: r["ranking"] for r in jl(s3.OUT / f"retrieval_{v}.jsonl")}
        ids = {r["instance_id"] for r in rows}
        gold = {t: s3.gold_for(truths_e[t], s3.load_index(tasks_e[t])) for t in ids}
        hits["external"][v] = key_hits(rows, {t: retr[t] for t in ids}, gold)
    for ds in hits:
        keys[ds] = {v: key_block(h) for v, h in hits[ds].items()}
        for v, h in hits[ds].items():
            keys[ds][v]["_n"] = len(h["_tids"])
    summary["keys"] = keys

    # ---- per-repo key effect, external
    per_repo: dict = {}
    for v, h in hits["external"].items():
        tids = h["_tids"]
        repos = np.array([tasks_e[t]["repo"] for t in tids])
        per_repo[v] = {}
        for repo in sorted(set(repos)):
            m = repos == repo
            e = {"n": int(m.sum())}
            for c in ("flat", "dichotomous"):
                a, b = h[c]["unit@1"][m], h["retrieval"]["unit@1"][m]
                e[c] = {"key_hits": int(a.sum()), "retrieval_hits": int(b.sum()),
                        "key_only": int((a > b).sum()), "retrieval_only": int((b > a).sum()),
                        "diff": round(float(a.mean() - b.mean()), 3)}
            per_repo[v][repo] = e
    summary["per_repo_external"] = per_repo
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    NOTE.write_text(render(summary))
    print(NOTE.read_text())


def fmt(e: dict) -> str:
    return f"{e['diff']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}] ({e['first_only']}/{e['second_only']})"


def render(s: dict) -> str:
    L = ["# S5 statistics: paired bootstrap CIs (2026-10-24)", "",
         "Reproduce: `source scripts/env.sh && uv run --no-sync python -m experiments.s5_stats` (reads saved per-task hits and key rankings, no models).",
         "Output: `results/s5_stats/summary.json`. Paired bootstrap over tasks, 5,000 resamples, seed 0, percentile 95% CI.",
         "Cell = first minus second [95% CI] (discordant pairs: first only / second only). Hit rates at k, n as in the S2/S3 tables.", ""]
    L += ["## 4.2 Retrieval baselines", ""]
    for ds, n in (("train", "n = 128 file, 121 unit"), ("external", "n = 153 file, 151 unit")):
        L += [f"### {ds} ({n})", "", "| comparison | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        names = sorted({k.split(" | ")[0] for k in s["retrieval"][ds]}, key=lambda x: [f"{a} - {b}" for a, b in RETRIEVAL_PAIRS].index(x))
        for nm in names:
            x, y = nm.split(" - ")
            L.append(f"| {LABEL[x]} - {LABEL[y]} | " + " | ".join(fmt(s["retrieval"][ds][f"{nm} | {m}"]) if f"{nm} | {m}" in s["retrieval"][ds] else "" for m in METRICS) + " |")
        L.append("")
    L += ["## 4.3 Repaired graph, leave-one-repo-out hybrid minus BM25 (train)", "", "| comparison | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for h in ("hybrid_rep_all_loro", "hybrid_old_callgraph_loro"):
        L.append(f"| {h} - BM25 | " + " | ".join(fmt(s["graph_loro"][f"{h} - bm25_units | {m}"]) for m in METRICS) + " |")
    L += ["", "## 5 Keys (unit@1, unit@10, file@1)", ""]
    for ds in ("train", "external"):
        for v in ("hybrid", "bm25"):
            blk = s["keys"][ds][v]
            L += [f"### {ds}, {v} candidates (n = {blk['_n']})", "", "| comparison | unit@1 | unit@10 | file@1 |", "|---|---|---|---|"]
            for c in ("flat - dichotomous", "flat - retrieval", "dichotomous - retrieval"):
                L.append(f"| {c} | " + " | ".join(fmt(blk[f"{c} | {m}"]) for m in ("unit@1", "unit@10", "file@1")) + " |")
            L.append("")
    L += ["## Per-repository key effect, external unit@1", "",
          "Counts only: key-only / retrieval-only tasks (net = key-only minus retrieval-only). n per repo is 9 to 22, so no per-repo CI is meaningful.", ""]
    for v in ("hybrid", "bm25"):
        L += [f"### {v} candidates", "", "| repo | n | retrieval hits | flat hits | flat only / ret only | dich hits | dich only / ret only |", "|---|---|---|---|---|---|---|"]
        for repo, e in s["per_repo_external"][v].items():
            f, d = e["flat"], e["dichotomous"]
            L.append(f"| {repo} | {e['n']} | {f['retrieval_hits']} | {f['key_hits']} | {f['key_only']} / {f['retrieval_only']} | {d['key_hits']} | {d['key_only']} / {d['retrieval_only']} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
