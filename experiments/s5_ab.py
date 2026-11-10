"""S5: leaderboard A/B of v1_budget (control) against v2_skill (+ repo-localization skill), Section 6.

Usage: source scripts/env.sh && uv run --no-sync python -m experiments.s5_ab
Reads the submission table in submission/LOG.md; rows still "in attesa" are skipped. Every scored row of an
arm's bundle hash counts, including the v1_budget submission made before the alternation started (same bundle).
Writes results/s5_ab/summary.json and paper/notes/s5_ab.md. Unpaired bootstrap, 5,000 resamples, seed 0.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import PermutationMethod, mannwhitneyu

B, SEED = 5000, 0
LB_TASKS = 58
LOG = Path("submission/LOG.md")
OUT = Path("results/s5_ab")
NOTE = Path("paper/notes/s5_ab.md")
ARMS = ("v1_budget", "v2_skill")  # control first; differences are reported as v2 minus v1


def load_scores(log: Path = LOG) -> dict[str, list[tuple[str, float]]]:
    """-> {bundle: [(date UTC, public LB score), ...]} for scored rows of the submission table."""
    out: dict[str, list[tuple[str, float]]] = {}
    for line in log.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not re.match(r"\d{4}-\d{2}-\d{2}", cells[0]):
            continue
        bundle, score = cells[2].strip("`"), cells[5].replace(",", ".")
        if re.fullmatch(r"\d+(\.\d+)?", score):
            out.setdefault(bundle, []).append((cells[0], float(score)))
    return out


def boot_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean of b minus mean of a, each arm resampled independently."""
    rng = np.random.default_rng(SEED)
    da = a[rng.integers(0, len(a), (B, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (B, len(b)))].mean(axis=1)
    d = db - da
    return {"diff": round(float(b.mean() - a.mean()), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}


def main() -> None:
    scores = load_scores()
    arms = {k: np.array([s for _, s in scores.get(k, [])]) for k in ARMS}
    a, b = arms[ARMS[0]], arms[ARMS[1]]
    summary: dict = {"arms": {}}
    for k, v in arms.items():
        summary["arms"][k] = {"n": int(len(v)), "scores": [s for _, s in scores.get(k, [])],
                              "mean": round(float(v.mean()), 4) if len(v) else None,
                              "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None}
    if len(a) >= 2 and len(b) >= 2:
        summary["v2 - v1"] = boot_diff(a, b)
        summary["v2 - v1"]["tasks"] = round(summary["v2 - v1"]["diff"] * LB_TASKS, 1)
        mw = mannwhitneyu(b, a, alternative="two-sided", method=PermutationMethod())  # exact with ties at these n
        summary["mann_whitney"] = {"U": float(mw.statistic), "p": round(float(mw.pvalue), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [f"# S5 leaderboard A/B ({date.today().isoformat()})", "",
             "Reproduce: `source scripts/env.sh && uv run --no-sync python -m experiments.s5_ab` (reads `submission/LOG.md`).",
             f"Public LB, {LB_TASKS} tasks (one task = {1 / LB_TASKS:.3f}). Unpaired bootstrap of the difference of means, "
             f"{B:,} resamples, seed {SEED}, percentile 95% CI; two-sided Mann-Whitney U, exact permutation p (ties present).", "",
             "| arm | n | scores | mean | SD |", "|---|---|---|---|---|"]
    for k, s in summary["arms"].items():
        fmt = ", ".join(f"{x:.2f}" for x in s["scores"])
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "-"
        sd = f"{s['sd']:.3f}" if s["sd"] is not None else "-"
        lines.append(f"| `{k}` | {s['n']} | {fmt} | {mean} | {sd} |")
    if "v2 - v1" in summary:
        d, mw = summary["v2 - v1"], summary["mann_whitney"]
        lines += ["", f"`v2_skill` minus `v1_budget`: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] "
                      f"({d['tasks']:+.1f} tasks); Mann-Whitney U = {mw['U']:.1f}, p = {mw['p']:.3f}."]
    NOTE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
