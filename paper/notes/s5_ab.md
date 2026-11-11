# S5 leaderboard A/B (2026-11-11)

Reproduce: `source scripts/env.sh && uv run --no-sync python -m experiments.s5_ab` (reads `submission/LOG.md`).
Public LB, 58 tasks (one task = 0.017). Unpaired bootstrap of the difference of means, 5,000 resamples, seed 0, percentile 95% CI; two-sided Mann-Whitney U, exact permutation p (ties present).

| arm | n | scores | mean | SD |
|---|---|---|---|---|
| `v1_budget` | 6 | 0.20, 0.18, 0.15, 0.13, 0.13, 0.12 | 0.152 | 0.032 |
| `v2_skill` | 5 | 0.18, 0.13, 0.15, 0.15, 0.17 | 0.156 | 0.019 |

`v2_skill` minus `v1_budget`: +0.004 [-0.023, +0.031] (+0.2 tasks); Mann-Whitney U = 17.5, p = 0.745.
