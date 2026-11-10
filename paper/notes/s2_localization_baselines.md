# S2 localization baselines (2026-09-28)

Reproduce: `uv run python -m experiments.s2_localize` (needs `experiments/s1_groundtruth.py` first).
Outputs: `results/s2_localize/{per_task.jsonl,summary.json}`. 129 train tasks; 128 with a gold
file present at base_commit, 121 with a gold function/class unit (module-level and new-file edits
have no unit). Candidates: every function, async function, method and class in non-test `.py`
files, indexed from the snapshot itself (not from the provided graph). Query: problem statement
only (the train set has no `hints_text`).

| method | file@1 | file@5 | file@10 | unit@1 | unit@5 | unit@10 | unit@20 | unit recall@10 |
|---|---|---|---|---|---|---|---|---|
| BM25 over files | .352 | .648 | .820 | | | | | |
| BM25 over units | .336 | .727 | .820 | .157 | .380 | .529 | .603 | .378 |
| PPR (BM25 seeds, own call graph) | .375 | .719 | .852 | .124 | .388 | .504 | .612 | .356 |
| Qwen3-Embedding-0.6B (local) | .484 | .766 | .852 | .240 | .471 | .595 | .702 | .457 |
| RRF(embedding, BM25) | .422 | .828 | .914 | .165 | .446 | .636 | .752 | .499 |
| provided `search_similar_code` | .195 | .266 | .328 | .107 | .124 | .165 | .207 | .117 |

"@k" = at least one gold item in the top k (hit rate). File rankings for unit methods take the
first occurrence of each file in the unit ranking.

Notes
- BM25 tokenization splits snake_case and CamelCase; path and qualname are up-weighted.
- PPR over our own name-resolved call graph (callee names with ≤5 candidates, plus class↔method
  edges) does not beat plain BM25 at the unit level; it helps slightly at file@10.
- 189,605 units across 127 commits collapse to 7,013 unique texts: embedding cost is small.
- Provided tool uses heuristic identifier queries from the issue (see S1 caveat); mapped to our
  unit ids via dotted module suffixes.
- n is small (121–128): differences under ~5 points are within noise. Bootstrap CIs to do in S5.
