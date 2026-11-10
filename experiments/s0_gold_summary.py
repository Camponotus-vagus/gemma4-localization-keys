"""Summarize a gold-patch check run: validity per repo and failure causes.

A task is valid for local evaluation when the gold patch resolves it and the empty patch does not.
Usage: uv run python experiments/s0_gold_summary.py results/s0_gold_all
Writes <run>/valid_tasks.txt with the valid instance ids.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PATTERNS = [
    r"ModuleNotFoundError: No module named '([^']+)'",
    r"(fixture '\w+' not found)",
    r"(recursive dependency involving fixture '\w+')",
    r"AttributeError: (module '\w+' has no attribute '\w+')",
    r"^E\s+(\w+Error)",
]


def cause_of(log: str) -> str:
    for pat in PATTERNS:
        if m := re.search(pat, log, re.M):
            return m.group(1)
    return "no error line"


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "gold_check.jsonl").read_text().splitlines()]
    gold = {r["instance_id"]: r for r in rows if r["mode"] == "gold"}
    empty = {r["instance_id"]: r for r in rows if r["mode"] == "empty"}

    per_repo: Counter[tuple[str, str]] = Counter()
    causes: Counter[str] = Counter()
    example: dict[str, str] = {}
    valid: list[str] = []

    for tid, g in sorted(gold.items()):
        e = empty.get(tid)
        if e is None:
            continue
        repo = g["repo"].split("/")[1]
        if g["resolved"] and not e["resolved"]:
            valid.append(tid)
            per_repo[(repo, "valid")] += 1
            continue
        per_repo[(repo, "invalid")] += 1
        if g["resolved"]:
            key = f"{repo}: empty patch also passes"
        else:
            log_path = run / "test_outputs" / f"{tid}.gold.log"
            log = log_path.read_text(errors="ignore") if log_path.exists() else ""
            key = f"{repo}: {cause_of(log)} (exit {g['exit_code']})"
        causes[key] += 1
        example.setdefault(key, tid)

    repos = sorted({r for r, _ in per_repo})
    print("repo       valid  invalid")
    for r in repos:
        print(f"{r:<10} {per_repo[(r, 'valid')]:>5}  {per_repo[(r, 'invalid')]:>7}")
    print(f"total      {len(valid):>5}  {sum(v for (_, s), v in per_repo.items() if s == 'invalid'):>7}\n")
    print("Causes of invalid tasks:")
    for key, n in causes.most_common():
        print(f"{n:>4}  {key}  e.g. {example[key]}")

    (run / "valid_tasks.txt").write_text("\n".join(valid) + "\n")


if __name__ == "__main__":
    main()
