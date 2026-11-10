# S3c graph-expanded candidates for the identification key (2026-10-24)

Status: candidate sets and ceilings done (no LLM). LLM runs prepared, not started (the MLX server is busy with another job).

Code: `src/g4a/graph_key.py` (graph view, candidate builders, grouped-card flat key), `experiments/s3_graphkey.py`
(`retrieval | run | summary`, `--set train|external`). No existing file modified. Outputs: `results/s3_graphkey/`
(train, 121 tasks with a gold unit) and `results/s3_graphkey/external/` (SWE-bench Verified subset, 151 of 153 tasks with a gold unit).
Per-variant numbers for all 37 variants: `retrieval_summary.json` in each folder.

## Design

Setting: in-sandbox, BM25 only, repaired graph (S3a). The graph expands the candidate set that the S2 flat key picks from; it is not a score.
The LLM prompt, parser and final ranking (picks first, then candidate priority order, then the rest of BM25) are the S2 flat key.

- `bm25_kK`: top-K BM25 units. Same prompt as S2 bm25 flat (K=20).
- `g_mM_hH_ORDER_kK` (b): top-M BM25 seeds, then neighbours as separate cards, then BM25 filler up to K cards. Neighbours = units within H hops over
  calls, maybe_calls, inherits, references (both directions) and contains parent/children. Intermediate nodes: no modules, no tests.
  ORDER: `bm25` (neighbour BM25 score, hops pooled), `links` (nearer hop, then number of seeds linking to it, then BM25), `samefile`
  (nearer hop, then same file as a top-3 seed, then BM25).
- `gc_mM_kK` (c): seeds as full cards; each seed's 1-hop neighbours (order `links`) shown as one-line entries under it, grouped by relation
  (`calls`, `called by`, `parent`, `children`, `related`). Every entry is numbered and pickable, so K counts all of them; one-line entries cost
  about half a card in characters. Needs a different listing, so the flat key lives in `graph_key.flat_key`.
- Retrieval-only unit@1/5/10 is identical to BM25 for every variant by construction (seeds, M >= 10, come first in priority order); the
  table below reports ceilings instead. Mean prompt characters of the listing are the cost proxy (S2 flat K=20 is about 1,270 prompt tokens in total).

## Candidate ceilings (train, n=121 tasks with a gold unit)

Unit ceiling = fraction of tasks with a gold unit among the displayed candidates; file ceiling = same for the gold file (over the tasks with a gold file in the index).
"vs char-matched" = tasks gained/lost against the bm25 variant with the closest mean characters.

| variant | unit ceiling | file ceiling | chars | vs bm25 same K | vs char-matched bm25 |
|---|---|---|---|---|---|
| bm25_k20 | 73/121 = .603 | .769 | 3,071 | | |
| bm25_k22 | 74 = .612 | .785 | 3,366 | | |
| bm25_k25 | 75 = .620 | .802 | 3,817 | | |
| bm25_k30 | 77 = .636 | .810 | 4,553 | | |
| bm25_k40 | 86 = .711 | .860 | 6,055 | | |
| bm25_k50 | 89 = .736 | .884 | 7,537 | | |
| g_m10_h1_bm25_k20 | 74 = .612 | .752 | 3,080 | +4/-3 | |
| g_m10_h1_samefile_k20 | 76 = .628 | .744 | 3,112 | +7/-4 | |
| g_m15_h1_links_k20 | 75 = .620 | .769 | 3,092 | +4/-2 | |
| g_m10_h2_links_k20 | 73 = .603 | .744 | 3,090 | +4/-4 | |
| gc_m10_k20 | 74 = .612 | .752 | 2,460 | +4/-3 | |
| **g_m10_h1_bm25_k30** | **82 = .678** | .802 | 4,588 | +9/-4 | same |
| g_m10_h1_links_k30 | 81 = .669 | .793 | 4,567 | +8/-4 | same |
| g_m10_h1_samefile_k30 | 79 = .653 | .785 | 4,581 | +8/-6 | same |
| g_m10_h2_bm25_k30 | 80 = .661 | .785 | 4,602 | +8/-5 | same |
| g_m10_h2_links_k30 | 80 = .661 | .785 | 4,566 | +8/-5 | same |
| g_m15_h1_bm25_k30 | 82 = .678 | .810 | 4,576 | +8/-3 | same |
| g_m15_h1_links_k30 | 78 = .645 | .769 | 4,611 | +4/-3 | same |
| g_m15_h2_bm25_k30 | 78 = .645 | .802 | 4,583 | +3/-2 | same |
| **gc_m10_k30** | **81 = .669** | .793 | 3,385 | +8/-4 | bm25_k22: +11/-4 |
| gc_m15_k30 | 78 = .645 | .769 | 3,652 | +4/-3 | bm25_k25: +6/-3 |
| gc_m10_k40 | 82 = .678 | .810 | 4,414 | +6/-10 vs k40 | bm25_k30: +9/-4 |
| gc_m15_k40 | 84 = .694 | .818 | 4,602 | +7/-9 vs k40 | bm25_k30: +10/-3 |

Readings:
- The gain at equal card count is small: +1 to +5 tasks net (K=30: .636 -> .678 at best; K=20: .603 -> .628 at best with samefile, and
  .603 -> .620 for the others). Two hops never beat one hop at the same K and M (h2 <= h1 in every pair); allowing paths through test nodes changes nothing
  (`g_m10_h2_links_k30_tests` = .661). The "76% reachable at 2 hops" of S3a holds for an unbounded pool of 61-320 neighbours; with 10-20
  slots most of it is lost because the cheap orderings cannot rank the pool.
- The graph file ceiling goes down at K=20 (.769 -> .744-.752): neighbours cluster in the same files as the seeds, whereas BM25 filler reaches more files.
- A plain BM25 K=40 (.711, 6,055 chars) beats every graph variant at K<=30 on ceiling, at +33% characters. The graph pays off per character, not per card:
  `gc_m10_k30` has the ceiling of `g_m10_h1_links_k30` with 26% fewer characters, and `gc_m15_k40` (.694 at 4,602 chars) is 7 tasks above bm25_k30 at the same length.
- Ceilings are an upper bound: S2 flat reached .430 unit@1 on BM25-only candidates (ceiling .603 at K=20), i.e. 71% of reachable.

## Chosen variants (at most 2 LLM variants)

Selection used the 121 train tasks only, and it is post-hoc over 37 variants on the same tasks the ceilings are computed on (no tuning data
separate from the evaluation data; differences of 1-5 tasks are within noise). Rule: highest unit ceiling at card budget 30, M=10, one hop.
1. `g_m10_h1_bm25_k30` (b: neighbours as separate cards): .678 vs .636, +9/-4.
2. `gc_m10_k30` (c: neighbours as grouped one-line entries): .669, +8/-4 at the same K and 26% fewer characters (+11/-4 vs bm25_k22 at equal length).
Both share seeds, so the pair isolates the card format. Baselines to run: `bm25_k30` (new) and `bm25_k20` (reused from S2 on train; same prompt, greedy).
K=20 is skipped: no variant clears +3 net tasks there with a stable file ceiling.
Alternative if one slot should go to the better equal-length candidate: `gc_m15_k40` (.694 train; also the best char-matched variant externally).

## External check (ceilings only, SWE-bench Verified subset, n=151 tasks with a gold unit; not used for selection)

| variant | unit ceiling | file ceiling | chars | vs char-matched bm25 |
|---|---|---|---|---|
| bm25_k20 | 97/151 = .642 | .907 | 3,821 | |
| bm25_k22 | 101 = .669 | .907 | 4,207 | |
| bm25_k25 | 101 = .669 | .914 | 4,793 | |
| bm25_k30 | 105 = .695 | .934 | 5,761 | |
| bm25_k40 | 107 = .709 | .940 | 7,702 | |
| g_m10_h1_bm25_k30 | 105 = .695 | .881 | 6,019 | bm25_k30: +6/-6 |
| gc_m10_k30 | 101 = .669 | .868 | 4,174 | bm25_k22: +5/-5 |
| gc_m15_k30 | 104 = .689 | .901 | 4,541 | bm25_k25: +5/-2 |
| gc_m10_k40 | 103 = .682 | .868 | 5,361 | bm25_k30: +6/-8 |
| gc_m15_k40 | 107 = .709 | .907 | 5,687 | bm25_k30: +7/-5 |

The train ceiling gain does not replicate for the two chosen variants: net 0 against the same-K and the char-matched BM25 baselines. Only the
`gc_m15_*` variants stay ahead by 3 tasks at equal length (net +3 at k25, +2 at k30-matched). Prior for the LLM runs: a null or small effect;
the ceiling gain on train (<= +5 tasks) would need the key to pick the new candidate every time to show at unit@1.

## Commands (run when the MLX server is free; each needs Gemma 4 31B via `mlx_lm`, as S2)

```
source scripts/env.sh
# candidate sets (no LLM; external graphs are cached in data/external/graphs_repaired, a cold build takes ~40 min)
uv run python -m experiments.s3_graphkey retrieval
uv run python -m experiments.s3_graphkey --set external retrieval
# train: reuses bm25_k20 from results/s2_key_bm25 (121 rows), runs 3 variants
uv run python -m experiments.s3_graphkey run --variants bm25_k20 bm25_k30 g_m10_h1_bm25_k30 gc_m10_k30
uv run python -m experiments.s3_graphkey summary
# external (held-out confirmation); bm25_k20 reuses results/s3_external/keys.jsonl (variant bm25, flat) where present
uv run python -m experiments.s3_graphkey --set external run --variants bm25_k20 bm25_k30 g_m10_h1_bm25_k30 gc_m10_k30
uv run python -m experiments.s3_graphkey --set external summary
```
Smoke test first: add `--limit 3`. Resumable (rows keyed by task and variant in `keys.jsonl`); `--task-ids` restricts tasks; `--no-reuse` reruns bm25_k20.

Runtime estimate (s2_key flat: ~15 s/task at ~1.27k prompt tokens; assume time grows with prompt length, generation is ~30 tokens):
train 121 tasks, ~17 s for k30 listings and ~15 s for gc_m10_k30 (3.4k chars): bm25_k30 ~35 min, g_m10 ~35 min, gc ~30 min, about 1.7 h in total.
External 151 tasks, listings are ~25% longer (~18-20 s): up to ~50 min per variant if bm25_k20 is not already reused, about 3.3 h for four variants.
Model load and the MLX contention with the other job come on top.

## Results (to fill)

Train, 121 tasks, paired:

| condition | file@1 | file@3 | unit@1 | unit@3 | unit@5 | unit@10 | prompt tok | s/task |
|---|---|---|---|---|---|---|---|---|
| bm25_k20 (S2 flat, reused) | | | .430 | | | | 1,267 | 15 |
| bm25_k30 | | | | | | | | |
| g_m10_h1_bm25_k30 | | | | | | | | |
| gc_m10_k30 | | | | | | | | |

Paired bootstrap unit@1 (5,000): g_m10_h1_bm25_k30 - bm25_k30: ; gc_m10_k30 - bm25_k30: ; vs bm25_k20: .
External: same table, to fill. Decision rule: graph expansion is kept in the paper only if it beats bm25_k30 (or equal-length bm25_k20) on unit@1 with a CI above 0 on train
and the sign holds on external; otherwise report it as a negative result next to the PPR score mixing of S3a.
