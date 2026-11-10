"""Regenerate the paper figures (PDF + PNG) from results/.

Run from anywhere:  source scripts/env.sh && uv run python paper/figures/make_figures.py

Data provenance
- fig_coverage:     results/s3_graph/summary.json (analyze.coverage_by_kind). Nothing hard-coded.
- fig_localization: results/s2_localize/summary.json and results/s3_external/summary.json
                    (hit rate at k in {1, 5, 10, 20} for all four methods). For BM25 and RRF the full
                    k=1..20 curve is computed from the per-task rankings (s2_key_bm25/retrieval.jsonl,
                    s2_key/retrieval.jsonl, s3_external/retrieval_{bm25,hybrid}.jsonl) plus the gold
                    symbols in data/{derived,external}/groundtruth.jsonl; the curve is checked against
                    the summary at the four k. PPR and Qwen3 embedding rankings were not saved, so
                    their curves are straight lines through the four summary points (dashed).
- fig_keys:         per-task rankings in results/s2_key*/keys.jsonl, s3_external/keys.jsonl and the
                    retrieval files above; 95% CIs by paired bootstrap over tasks (5,000 resamples).
No number is taken from the notes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

MIN_EXTERNAL_TASKS = 140
N_BOOT = 5000
KS = np.arange(1, 21)
SUMMARY_KS = (1, 5, 10, 20)

# Okabe-Ito, one color per method, reused across figures.
C = {
    "BM25": "#0072B2",
    "PPR": "#E69F00",
    "Qwen3 embedding": "#009E73",
    "RRF": "#D55E00",
    "provided": "#9A9A9A",
    "repaired": "#2B2B2B",
    "retrieval": "#9A9A9A",
    "flat": "#56B4E9",
    "dichotomous": "#CC79A7",
}
INK, MUTED, GRID = "#222222", "#666666", "#E3E3E3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def gold_units(truth_path: Path) -> dict[str, set[str]]:
    """Gold function/class units per task. A gold symbol absent from the index never appears in a
    ranking, so using the unfiltered set gives the same hit values as filtering by the index."""
    out: dict[str, set[str]] = {}
    for r in load_jsonl(truth_path):
        out[r["instance_id"]] = {f"{s['path']}::{s['qualname']}" for s in r["symbols"] if s["kind"] != "module"}
    return out


def hit_curve(rankings: dict[str, list[str]], gold: dict[str, set[str]], tasks: list[str]) -> np.ndarray:
    """Hit rate at k=1..20 over the given tasks."""
    hits = np.zeros((len(tasks), len(KS)))
    for i, t in enumerate(tasks):
        pos = [j for j, u in enumerate(rankings[t][:20]) if u in gold[t]]
        if pos:
            hits[i, pos[0]:] = 1
    return hits.mean(axis=0)


# ---------------------------------------------------------------- fig 1

KIND_LABELS = [
    ("sync_function", "Sync function / method"),
    ("class", "Class"),
    ("async_function", "Async function / method"),
    ("module_level", "Module-level edit"),
    ("new_file", "New file"),
]


def fig_coverage() -> None:
    cov = load_json(RES / "s3_graph" / "summary.json")["analyze"]["coverage_by_kind"]
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    h = 0.36
    y = np.arange(len(KIND_LABELS))[::-1].astype(float)
    for off, key, label in ((+h / 2, "provided", "Provided graph"), (-h / 2, "repaired", "Repaired graph")):
        for yi, (kind, _) in zip(y, KIND_LABELS):
            n, c = cov[kind]["n"], cov[kind][key]
            pct = 100 * c / n
            ax.barh(yi + off, pct, height=h * 0.92, color=C[key], label=label if kind == KIND_LABELS[0][0] else None)
            ax.text(pct + 1.5, yi + off, f"{c}/{n}  ({pct:.0f}%)", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(y, [lab for _, lab in KIND_LABELS])
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Gold symbols present as graph nodes (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig_coverage")


# ---------------------------------------------------------------- fig 2

METHODS = [  # (label, key in summary.json)
    ("BM25", "bm25_units"),
    ("PPR", "ppr"),
    ("Qwen3 embedding", "emb"),
    ("RRF", "emb+bm25"),
]


def fig_localization() -> None:
    sets = {
        "Train (competition, n=121)": dict(
            summ=RES / "s2_localize" / "summary.json",
            truth=ROOT / "data/derived/groundtruth.jsonl",
            ranks={"BM25": RES / "s2_key_bm25/retrieval.jsonl", "RRF": RES / "s2_key/retrieval.jsonl"},
        ),
        "External (SWE-bench Verified, n=151)": dict(
            summ=RES / "s3_external" / "summary.json",
            truth=ROOT / "data/external/groundtruth.jsonl",
            ranks={"BM25": RES / "s3_external/retrieval_bm25.jsonl", "RRF": RES / "s3_external/retrieval_hybrid.jsonl"},
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, (title, s) in zip(axes, sets.items()):
        summ = load_json(s["summ"])
        n = summ["bm25_units"]["n_units"]
        gold = gold_units(s["truth"])
        for label, key in METHODS:
            pts = np.array([summ[key][f"unit_hit@{k}"] for k in SUMMARY_KS])
            if label in s["ranks"]:
                rk = {r["instance_id"]: r["ranking"] for r in load_jsonl(s["ranks"][label])}
                # tasks with at least one gold function/class unit; the summary's n_units is the same set
                # restricted to units present in the index
                tasks = [t for t in rk if gold.get(t)]
                curve = hit_curve(rk, gold, tasks)
                # consistency check against the summary (tolerance: summary rounds to 3 decimals; the
                # task set can differ by the few tasks whose gold unit is not in the index)
                got = curve[[k - 1 for k in SUMMARY_KS]]
                scale = len(tasks) / n
                if len(tasks) != n:
                    print(f"  note: {title} {label}: {len(tasks)} tasks with gold units vs n_units={n}")
                if not np.allclose(got * scale, pts, atol=0.006):
                    print(f"  note: {title} {label}: recomputed {np.round(got * scale, 3)} vs summary {np.round(pts, 3)} (n={len(tasks)} vs {n})")
                curve = curve * scale
                ax.plot(KS, curve, color=C[label], lw=1.6, label=label)
            else:
                ax.plot(SUMMARY_KS, pts, color=C[label], lw=1.3, ls=(0, (4, 2)), label=label)
            ax.plot(SUMMARY_KS, pts, "o", color=C[label], ms=3.2, mec="white", mew=0.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, 20.5)
        ax.set_xlabel("k (candidates inspected)")
        ax.text(0.03, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=MUTED)
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("unit@k (gold unit in top k)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig_localization")


# ---------------------------------------------------------------- fig 3

def boot_ci(hit: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    means = hit[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_hits(
    cands: dict[str, dict],  # candidate source -> {retrieval path, keys path rows}
    gold: dict[str, set[str]],
) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    """unit@1 per task for every (candidate source, condition), restricted to tasks complete for all."""
    top1: dict[tuple[str, str], dict[str, int]] = {}
    for src, d in cands.items():
        top1[(src, "retrieval")] = {r["instance_id"]: int(bool(set(r["ranking"][:1]) & gold[r["instance_id"]])) for r in d["retrieval"]}
        for r in d["keys"]:
            top1.setdefault((src, r["mode"]), {})[r["instance_id"]] = int(bool(set(r["ranking"][:1]) & gold[r["instance_id"]]))
    key_conds = [c for c in top1 if c[1] != "retrieval"]
    tasks = sorted(set.intersection(*(set(top1[c]) for c in key_conds)))
    tasks = [t for t in tasks if all(t in top1[c] for c in top1)]
    return tasks, {c: np.array([top1[c][t] for t in tasks]) for c in top1}


def key_data_train() -> tuple[list[str], dict]:
    gold = gold_units(ROOT / "data/derived/groundtruth.jsonl")
    cands = {
        "hybrid": dict(retrieval=load_jsonl(RES / "s2_key/retrieval.jsonl"), keys=load_jsonl(RES / "s2_key/keys.jsonl")),
        "bm25": dict(retrieval=load_jsonl(RES / "s2_key_bm25/retrieval.jsonl"), keys=load_jsonl(RES / "s2_key_bm25/keys.jsonl")),
    }
    return paired_hits(cands, gold)


def key_data_external() -> tuple[list[str], dict]:
    gold = gold_units(ROOT / "data/external/groundtruth.jsonl")
    keys = load_jsonl(RES / "s3_external/keys.jsonl")
    cands = {
        v: dict(
            retrieval=load_jsonl(RES / f"s3_external/retrieval_{v}.jsonl"),
            keys=[{**r} for r in keys if r["variant"] == v],
        )
        for v in ("hybrid", "bm25")
    }
    return paired_hits(cands, gold)


def fig_keys() -> None:
    panels = {"Train (competition)": key_data_train()}
    ext_path = RES / "s3_external" / "keys.jsonl"
    if ext_path.exists():
        tasks, hits = key_data_external()
        if len(tasks) >= MIN_EXTERNAL_TASKS:
            panels["External (SWE-bench Verified)"] = (tasks, hits)
        else:
            print(f"external panel skipped: {len(tasks)} complete tasks < {MIN_EXTERNAL_TASKS}")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 if len(panels) > 1 else 3.4, 2.8), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    conds = [("retrieval", "Retrieval only"), ("flat", "Flat key"), ("dichotomous", "Dichotomous key")]
    groups = [("hybrid", "Hybrid candidates"), ("bm25", "BM25 candidates")]
    w = 0.26
    for ax, (title, (tasks, hits)) in zip(axes[0], panels.items()):
        n = len(tasks)
        idx = rng.integers(0, n, (N_BOOT, n))  # one resample shared by all conditions: paired
        for gi, (src, _) in enumerate(groups):
            for ci, (cond, label) in enumerate(conds):
                h = hits[(src, cond)]
                m = h.mean()
                lo, hi = boot_ci(h, idx)
                x = gi + (ci - 1) * (w + 0.03)
                ax.bar(x, m, width=w, color=C[cond], label=label if gi == 0 else None)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.9, capsize=2, capthick=0.9)
                ax.text(x, hi + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
            d = hits[(src, "flat")] - hits[(src, "dichotomous")]
            bd = d[idx].mean(axis=1)
            print(f"  [{title}] {src}: n={n}, flat-dichotomous {d.mean():+.3f} "
                  f"[{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
        ax.set_xticks(range(len(groups)), [g[1] for g in groups])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 0.88)
        ax.text(0.02, 0.98, f"{title}, n={n}", transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=MUTED)
    axes[0][0].set_ylabel("unit@1")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_keys")


if __name__ == "__main__":
    fig_coverage()
    fig_localization()
    fig_keys()
