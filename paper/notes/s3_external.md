# S3b external generalization: SWE-bench Verified subset (2026-10-24)

Question: do the S2 localization findings hold on public repos the competition never touched?

Reproduce (all from the project root, after `source scripts/env.sh`; Python via `uv run --no-sync`):
```
# dataset: HF princeton-nlp/SWE-bench_Verified (test) -> data/external/swebench_verified.jsonl (parquet, 500 rows)
# bare partial clones of the 8 GitHub repos: git clone --filter=blob:none --bare https://github.com/<org>/<repo>.git data/external/repos/<repo>.git
uv run python -m experiments.s3_external select      # -> experiments/s3_external_tasks.json (committed)
uv run python -m experiments.s3_external snapshots   # git archive of *.py at base_commit -> data/external/snapshots/<id>.tar.gz
uv run python -m experiments.s3_external truth       # gold `patch` -> data/external/groundtruth.jsonl, prints kind table
uv run python -m experiments.s3_external baselines   # -> results/s3_external/{per_task.jsonl,summary.json}
uv run python -m experiments.s3_external baseline-summary   # overall + per-repo tables
uv run python -m experiments.s3_external retrieval   # top-200 rankings (hybrid, bm25)
uv run python -m experiments.s3_external key         # 4 LLM conditions, resumable -> results/s3_external/keys.jsonl
uv run python -m experiments.s3_external summary     # paired table + bootstrap
```
Code: `src/g4a/external.py` (selection, snapshots, ground truth without a provided graph), `experiments/s3_external.py`.
Retrieval, indexing, metrics and the key prompts are the S2 ones (`g4a.localize`, `g4a.embed_rank`, `g4a.key_localize`), unchanged.

## Task selection

153 tasks, seed 20261024 for the sampled repos. `psf/requests` (train) excluded; sympy, seaborn, flask not used.

| repo | tasks | selection |
|---|---|---|
| pytest-dev/pytest | 19 | all |
| pydata/xarray | 22 | all |
| pylint-dev/pylint | 10 | all |
| astropy/astropy | 22 | all |
| sphinx-doc/sphinx | 20 | sample of 44 |
| scikit-learn/scikit-learn | 20 | sample of 32 |
| matplotlib/matplotlib | 20 | sample of 34 |
| django/django | 20 | sample of 231 |

Differences from the train setting, all else equal: the query is the problem statement only (no `hints_text`);
snapshots hold only `*.py` files (no `.git`, no docs), which is all the index reads; gold comes from the dataset `patch`
(source edits only, tests are in `test_patch`). Candidate files exclude test paths by the S1 regex, so e.g.
`lib/matplotlib/testing/` and `astropy/**/tests/` are not candidates (and never gold). Embeddings: Qwen3-Embedding-0.6B in
fp16 on MPS (2x faster than fp32; cosine to the fp32 vectors 0.9999+), cache in `data/external/emb_cache/`.

## Ground truth

Same code as S1 (`parse_patch`, `enclosing_symbols`), applied to the pre-fix source. No graph exists for these repos, so
there is no in-graph column; graph coverage will come from our own builder (S3 graph work).

| symbol kind | n |
|---|---|
| sync function / method | 263 |
| class | 30 |
| async function / method | 0 |
| module-level edit | 44 |
| new file | 1 |
| total | 338 |

153/153 tasks have a gold file present at base_commit; 151/153 have a gold function/class unit (one pylint and one django
task only edit module-level code). Per repo with a gold unit: all tasks except pylint 9/10, django 19/20. The gold set is
mostly single-function (SWE-bench Verified filters for well-specified, small fixes); async edits do not occur, so the
async hole of the provided graphs cannot be measured here.

## Baselines (no LLM)

Hit rate (at least one gold item in top k). n = 153 for file metrics, 151 for unit metrics.

| method | file@1 | file@5 | file@10 | unit@1 | unit@5 | unit@10 | unit@20 | unit recall@10 |
|---|---|---|---|---|---|---|---|---|
| BM25 over files | .399 | .719 | .837 | | | | | |
| BM25 over units | .477 | .771 | .876 | .199 | .450 | .550 | .642 | .462 |
| PPR (BM25 seeds, own call graph) | .477 | .765 | .863 | .166 | .391 | .523 | .649 | .448 |
| Qwen3-Embedding-0.6B | .647 | .869 | .928 | .272 | .563 | .649 | .735 | .551 |
| RRF(embedding, BM25) | .634 | .902 | .974 | .305 | .603 | .689 | .801 | .579 |

Per-repo numbers (n of 9-22) are in `uv run python -m experiments.s3_external baseline-summary`; they swing by 20-40
points between repos (xarray is the hard one for BM25: file@1 .14 over files, .27 over units; embeddings reach .55), so
treat repo-level differences as noise.

Against the train numbers (S2, n = 121-128), in words:
- The ordering reproduces: RRF is best at unit@5 and beyond, the embedding is best or tied at @1 among single signals,
  BM25 over units is the weakest dense-free baseline at file level, and PPR over our call graph again does not beat plain BM25
  at unit@1/@5 (it ties at unit@20).
- Absolute levels are higher than on train, most clearly at file@1 (BM25 units .336 -> .477, embedding .484 -> .647,
  RRF .422 -> .634) and unit@1 for RRF (.165 -> .305). Plausible reasons: SWE-bench issues are longer and more explicit
  about identifiers and tracebacks than the private-style competition tasks; the repos are older and widely written about.
  The sets differ in both repos and task source, so this is a shift, not a paired comparison.
- RRF unit@1 (.305) is no longer clearly above the embedding alone (.272), and the file@1 of RRF (.634) is not above the
  embedding (.647); in S2 RRF was ahead at both. The gain from fusion is at @5/@10/@20.

## Identification keys (Gemma 4 31B MLX 4-bit, thinking off, greedy, 200 new tokens)

Smoke test (5 astropy tasks, all four conditions, 2026-10-24): all conditions run and parse; flat/hybrid unit@1 5/5,
dichotomous/hybrid 4/5, dichotomous/bm25 4/5, flat/bm25 3/5. Prompts are longer than in S2 (flat 1,500-3,100 tokens,
dichotomous 2,700-5,100), ~35-45 s per flat call, ~50-70 s per dichotomous call. One bm25/dichotomous task took 131 s and
returned no pick (output cut at 200 tokens); this is a failure mode to count in the summary. n = 5 means nothing statistically.

Full run (2026-10-24, 151 paired tasks with a gold unit; `uv run python -m experiments.s3_external summary`, diagnostics
`uv run python -m experiments.s3_external_diag` -> `results/s3_external/diag.json`):

| condition | file@1 | file@3 | unit@1 | unit@3 | unit@5 | unit@10 | calls | prompt tok | s/task |
|---|---|---|---|---|---|---|---|---|---|
| hybrid retrieval only | .636 | .868 | .305 | .517 | .603 | .689 | 0 | 0 | 0 |
| hybrid + flat | .861 | .914 | .623 | .715 | .735 | .768 | 1 | 1,997 | 22 |
| hybrid + dichotomous | .828 | .921 | .616 | .695 | .722 | .755 | 2 | 3,331 | 38 |
| BM25 retrieval only | .483 | .675 | .199 | .364 | .450 | .550 | 0 | 0 | 0 |
| BM25 + flat | .788 | .854 | .517 | .596 | .603 | .636 | 1 | 1,993 | 22 |
| BM25 + dichotomous | .755 | .821 | .510 | .596 | .609 | .669 | 2 | 3,278 | 38 |

Paired bootstrap (5,000 resamples), unit@1, discordant pairs as (first better vs second better):
- hybrid: flat - dichotomous +.007 [-.060, +.073] (14 vs 13); flat - retrieval +.318 [+.225, +.411] (56 vs 8); dichotomous - retrieval +.311 [+.219, +.404] (56 vs 9).
- BM25: flat - dichotomous +.007 [-.053, +.073] (12 vs 11); flat - retrieval +.318 [+.238, +.397] (51 vs 3); dichotomous - retrieval +.311 [+.232, +.391] (51 vs 4).

Answers: the flat > dichotomous advantage on hybrid candidates (S2: +.099 [+.025, +.174]) does not replicate (+.007); on BM25
candidates the S2 sign flip is gone too (S2 -.017, here +.007), all within noise. The key effect over retrieval replicates: flat +.318 here against +.413 on train, dichotomous +.311 against +.314. The flat gain is smaller because external retrieval is stronger (unit@1 .305 against .165). At equal accuracy the flat key uses one call and about 40% fewer prompt tokens (1,997 vs 3,331;
1,993 vs 3,278) and 22 s against 38 s per task.

Key-quality diagnostics (`experiments/s3_external_diag.py`; train recomputed with the same script). First couplet wrong =
none of the (at most 2) chosen files holds a gold unit; the fallback to the top file applies when nothing parses.

| | ext hybrid | ext BM25 | train hybrid | train BM25 |
|---|---|---|---|---|
| tasks | 151 | 151 | 121 | 121 |
| dichotomous first couplet wrong | 25 | 36 | 30 | 37 |
| of those, flat unit@1 right | 9 | 6 | 11 | 7 |
| dichotomous unit@1 right among them | 0 | 0 | 0 | 0 |
| flat ceiling: gold unit absent from the 20 cards | 30 (.801 reachable) | 54 (.642) | 30 (.752) | 48 (.603) |
| flat unit@1 / reachable | .623 (78%) | .517 (81%) | .579 (77%) | .430 (71%) |
| gold file absent from the 10 files | 3 | 18 | 8 | 20 |
| flat, no pick parsed | 0 | 1 | 0 | 8 |
| dichotomous, no unit pick (second couplet) | 2 | 12 | 3 | 5 |
| dichotomous, first couplet unparsed | 0 | 2 | 1 | 0 |

Scoring of unparsed output: a flat call with no pick returns the retrieval order unchanged (unit@1 of the retrieval; 0 of 1
here, 0 of 8 on train BM25). A dichotomous second couplet with no pick returns the units of the chosen files in retrieval order
(unit@1 right in 1 of 2 hybrid, 1 of 12 BM25 on ext). A first couplet that does not parse falls back to the top file. No task
is dropped. Notes: the 30/121 first-couplet count in the S2 notes reproduces; the flat rescue count there (12) recomputes as 11.
With "gold file" instead of "file of a gold unit" the train hybrid figures are 28 and 10 (external figures do not change).
Reading: the dichotomous key errs at the first couplet in 17% of external tasks (hybrid) against 25% on train. Flat rescues 9 of the 25, yet the net is zero (14 vs 13 discordant pairs), so the flat key also loses tasks the dichotomous key gets right.
