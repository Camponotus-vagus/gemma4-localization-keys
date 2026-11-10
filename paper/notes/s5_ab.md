# S5 leaderboard A/B (2026-11-10)

Reproduce: `source scripts/env.sh && uv run --no-sync python -m experiments.s5_ab` (reads `submission/LOG.md`).
Public LB, 58 tasks (one task = 0.017). Unpaired bootstrap of the difference of means, 5,000 resamples, seed 0, percentile 95% CI; two-sided Mann-Whitney U, exact permutation p (ties present).

| arm | n | scores | mean | SD |
|---|---|---|---|---|
| `v1_budget` | 5 | 0.20, 0.18, 0.15, 0.13, 0.13 | 0.158 | 0.031 |
| `v2_skill` | 5 | 0.18, 0.13, 0.15, 0.15, 0.17 | 0.156 | 0.019 |

`v2_skill` minus `v1_budget`: -0.002 [-0.032, +0.026] (-0.1 tasks); Mann-Whitney U = 12.5, p = 1.000.
