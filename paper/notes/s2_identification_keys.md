# S2 identification-key localization (2026-09-28)

Reproduce: `uv run python -m experiments.s2_key retrieval && uv run python -m experiments.s2_key run && uv run python -m experiments.s2_key summary`
Outputs: `results/s2_key/`. Model: Gemma 4 31B-it, MLX 4-bit (`mlx-community/gemma-4-31b-it-4bit`),
thinking disabled, greedy, max 200 new tokens, on a MacBook Air M5 32 GB. 121 train tasks with a
gold unit. Candidates for both keys come from the same hybrid retrieval (RRF of Qwen3-Embedding-0.6B
and BM25 over units). The model's picks are placed first, then the retrieval order.

| condition | file@1 | file@3 | unit@1 | unit@3 | unit@5 | unit@10 | LLM calls | prompt tok | s/task |
|---|---|---|---|---|---|---|---|---|---|
| retrieval only | .438 | .719 | .165 | .397 | .446 | .636 | 0 | 0 | 0 |
| multi-access key (flat, top-20 units) | .785 | .826 | .579 | .620 | .636 | .694 | 1 | 1,241 | 14 |
| dichotomous key (10 files -> ≤2 files -> ≤15 units) | .752 | .835 | .479 | .612 | .661 | .727 | 2 | 1,953 | 23 |

Paired bootstrap (5,000 resamples, unit@1):
- flat - dichotomous: +0.099, 95% CI [+0.025, +0.174]; discordant pairs 18 vs 6.
- flat - retrieval: +0.413 [+0.322, +0.504]; dichotomous - retrieval: +0.314 [+0.215, +0.413].

Error propagation: the dichotomous key picks a wrong file at the first couplet in 30/121 tasks;
in 12 of those the flat key still ranks a gold unit first. An error at the first couplet cannot be
recovered later, which is the standard argument for multi-access over dichotomous keys in
taxonomy (robustness to a misread or missing character).

Candidate ceiling: the gold unit is absent from the flat key's 20 candidates in 30/121 tasks, so
flat unit@1 is capped at 91/121 = .752; it reaches .579 (77% of reachable). The gold file is absent
from the dichotomous key's 10 candidate files in 8/121 tasks.

Pre-registered hypothesis (dichotomous > flat retrieval) is NOT supported at @1; dichotomous is
slightly better at @5/@10 because units of the chosen files are promoted as a block.

Open questions
- Realistic in-sandbox setting: no embedding model inside the agent sandbox, so re-run both keys
  on BM25-only candidates.
- Larger candidate pools for the flat key (ceiling) vs prompt length.
- Multi-access key with explicit characters (e.g. "mentions error message", "is a public API")
  instead of free-form cards.

## BM25-only candidates (in-sandbox setting, 2026-09-28)

`uv run python -m experiments.s2_key --variant bm25 {retrieval,run,summary}` -> `results/s2_key_bm25/`.

| condition | file@1 | file@3 | unit@1 | unit@3 | unit@5 | unit@10 | calls | prompt tok | s/task |
|---|---|---|---|---|---|---|---|---|---|
| BM25 only | .347 | .587 | .157 | .281 | .380 | .529 | 0 | 0 | 0 |
| dichotomous | .702 | .785 | .446 | .545 | .579 | .603 | 2 | 1,965 | 23 |
| flat | .669 | .744 | .430 | .512 | .537 | .579 | 1 | 1,267 | 15 |

With weaker candidates the ordering flips (dichotomous slightly ahead) but within noise: paired
bootstrap unit@1 flat - dichotomous -0.017, 95% CI [-0.099, +0.058], discordant pairs 11 vs 13 (2026-10-24). Working hypothesis: the dichotomous key's file couplet reaches beyond
the flat key's 20 units (10 files cover more of the pool), which matters more when the unit
ranking is poor. The best key depends on retrieval quality, as key choice in taxonomy depends on
how reliable the characters are.
