# S5 statistics: paired bootstrap CIs (2026-10-24)

Reproduce: `source scripts/env.sh && uv run --no-sync python -m experiments.s5_stats` (reads saved per-task hits and key rankings, no models).
Output: `results/s5_stats/summary.json`. Paired bootstrap over tasks, 5,000 resamples, seed 0, percentile 95% CI.
Cell = first minus second [95% CI] (discordant pairs: first only / second only). Hit rates at k, n as in the S2/S3 tables.

## 4.2 Retrieval baselines

### train (n = 128 file, 121 unit)

| comparison | unit@1 | unit@10 | file@1 | file@10 |
|---|---|---|---|---|
| RRF - BM25 | +0.008 [-0.066, +0.083] (11/10) | +0.107 [+0.025, +0.198] (21/8) | +0.086 [+0.000, +0.180] (23/12) | +0.094 [+0.039, +0.156] (14/2) |
| Qwen3 - BM25 | +0.083 [+0.000, +0.165] (18/8) | +0.066 [-0.041, +0.174] (26/18) | +0.148 [+0.055, +0.234] (28/9) | +0.031 [-0.039, +0.102] (13/9) |
| PPR - BM25 | -0.033 [-0.099, +0.033] (7/11) | -0.025 [-0.083, +0.033] (5/8) | +0.039 [-0.047, +0.125] (18/13) | +0.031 [+0.008, +0.062] (4/0) |
| RRF - Qwen3 | -0.074 [-0.149, +0.000] (6/15) | +0.041 [-0.041, +0.124] (16/11) | -0.062 [-0.133, +0.008] (6/14) | +0.062 [+0.016, +0.109] (9/1) |
| provided - BM25 | -0.050 [-0.124, +0.025] (8/14) | -0.364 [-0.463, -0.265] (4/48) | -0.141 [-0.227, -0.055] (9/27) | -0.492 [-0.586, -0.398] (2/65) |

### external (n = 153 file, 151 unit)

| comparison | unit@1 | unit@10 | file@1 | file@10 |
|---|---|---|---|---|
| RRF - BM25 | +0.106 [+0.040, +0.172] (23/7) | +0.139 [+0.073, +0.205] (26/5) | +0.157 [+0.072, +0.242] (36/12) | +0.098 [+0.052, +0.150] (15/0) |
| Qwen3 - BM25 | +0.073 [-0.013, +0.152] (27/16) | +0.099 [+0.013, +0.185] (30/15) | +0.170 [+0.078, +0.261] (42/16) | +0.052 [+0.000, +0.105] (12/4) |
| PPR - BM25 | -0.033 [-0.080, +0.013] (4/9) | -0.026 [-0.073, +0.020] (4/8) | +0.000 [-0.059, +0.059] (10/10) | -0.013 [-0.033, +0.000] (0/2) |
| RRF - Qwen3 | +0.033 [-0.053, +0.119] (25/20) | +0.040 [-0.020, +0.099] (14/8) | -0.013 [-0.105, +0.078] (24/26) | +0.046 [+0.006, +0.085] (8/1) |

## 4.3 Repaired graph, leave-one-repo-out hybrid minus BM25 (train)

| comparison | unit@1 | unit@10 | file@1 | file@10 |
|---|---|---|---|---|
| hybrid_rep_all_loro - BM25 | -0.008 [-0.025, +0.000] (0/1) | +0.008 [+0.000, +0.025] (1/0) | -0.008 [-0.023, +0.000] (0/1) | +0.000 [+0.000, +0.000] (0/0) |
| hybrid_old_callgraph_loro - BM25 | -0.074 [-0.132, -0.025] (2/11) | -0.033 [-0.083, +0.008] (2/6) | -0.023 [-0.094, +0.047] (9/12) | +0.023 [+0.000, +0.055] (3/0) |

## 5 Keys (unit@1, unit@10, file@1)

### train, hybrid candidates (n = 121)

| comparison | unit@1 | unit@10 | file@1 |
|---|---|---|---|
| flat - dichotomous | +0.099 [+0.025, +0.174] (18/6) | -0.033 [-0.116, +0.041] (10/14) | +0.033 [-0.050, +0.107] (14/10) |
| flat - retrieval | +0.413 [+0.314, +0.504] (51/1) | +0.058 [+0.017, +0.107] (8/1) | +0.347 [+0.265, +0.430] (42/0) |
| dichotomous - retrieval | +0.314 [+0.215, +0.413] (43/5) | +0.091 [+0.008, +0.174] (19/8) | +0.314 [+0.215, +0.413] (44/6) |

### train, bm25 candidates (n = 121)

| comparison | unit@1 | unit@10 | file@1 |
|---|---|---|---|
| flat - dichotomous | -0.017 [-0.099, +0.058] (11/13) | -0.025 [-0.099, +0.050] (9/12) | -0.033 [-0.107, +0.041] (8/12) |
| flat - retrieval | +0.273 [+0.190, +0.356] (35/2) | +0.050 [+0.017, +0.091] (6/0) | +0.322 [+0.240, +0.405] (40/1) |
| dichotomous - retrieval | +0.289 [+0.207, +0.380] (37/2) | +0.074 [+0.000, +0.149] (15/6) | +0.355 [+0.273, +0.446] (44/1) |

### external, hybrid candidates (n = 151)

| comparison | unit@1 | unit@10 | file@1 |
|---|---|---|---|
| flat - dichotomous | +0.007 [-0.060, +0.073] (14/13) | +0.013 [-0.053, +0.080] (15/13) | +0.033 [-0.020, +0.086] (11/6) |
| flat - retrieval | +0.318 [+0.225, +0.411] (56/8) | +0.080 [+0.040, +0.126] (12/0) | +0.225 [+0.152, +0.305] (38/4) |
| dichotomous - retrieval | +0.311 [+0.218, +0.404] (56/9) | +0.066 [-0.007, +0.139] (22/12) | +0.192 [+0.113, +0.278] (39/10) |

### external, bm25 candidates (n = 151)

| comparison | unit@1 | unit@10 | file@1 |
|---|---|---|---|
| flat - dichotomous | +0.007 [-0.053, +0.073] (12/11) | -0.033 [-0.093, +0.026] (8/13) | +0.033 [-0.013, +0.080] (9/4) |
| flat - retrieval | +0.318 [+0.238, +0.397] (51/3) | +0.086 [+0.046, +0.133] (13/0) | +0.305 [+0.225, +0.384] (48/2) |
| dichotomous - retrieval | +0.311 [+0.232, +0.391] (51/4) | +0.119 [+0.060, +0.185] (23/5) | +0.272 [+0.192, +0.351] (45/4) |

## Per-repository key effect, external unit@1

Counts only: key-only / retrieval-only tasks (net = key-only minus retrieval-only). n per repo is 9 to 22, so no per-repo CI is meaningful.

### hybrid candidates

| repo | n | retrieval hits | flat hits | flat only / ret only | dich hits | dich only / ret only |
|---|---|---|---|---|---|---|
| astropy/astropy | 22 | 4 | 20 | 16 / 0 | 20 | 16 / 0 |
| django/django | 19 | 7 | 12 | 7 / 2 | 8 | 5 / 4 |
| matplotlib/matplotlib | 20 | 6 | 11 | 6 / 1 | 11 | 5 / 0 |
| pydata/xarray | 22 | 6 | 11 | 6 / 1 | 14 | 8 / 0 |
| pylint-dev/pylint | 9 | 0 | 5 | 5 / 0 | 5 | 5 / 0 |
| pytest-dev/pytest | 19 | 8 | 11 | 5 / 2 | 12 | 5 / 1 |
| scikit-learn/scikit-learn | 20 | 8 | 15 | 8 / 1 | 14 | 7 / 1 |
| sphinx-doc/sphinx | 20 | 7 | 9 | 3 / 1 | 9 | 5 / 3 |

### bm25 candidates

| repo | n | retrieval hits | flat hits | flat only / ret only | dich hits | dich only / ret only |
|---|---|---|---|---|---|---|
| astropy/astropy | 22 | 3 | 16 | 13 / 0 | 19 | 16 / 0 |
| django/django | 19 | 4 | 10 | 7 / 1 | 8 | 5 / 1 |
| matplotlib/matplotlib | 20 | 6 | 8 | 3 / 1 | 7 | 2 / 1 |
| pydata/xarray | 22 | 2 | 9 | 7 / 0 | 9 | 7 / 0 |
| pylint-dev/pylint | 9 | 2 | 4 | 2 / 0 | 4 | 2 / 0 |
| pytest-dev/pytest | 19 | 4 | 10 | 6 / 0 | 9 | 5 / 0 |
| scikit-learn/scikit-learn | 20 | 4 | 14 | 10 / 0 | 14 | 10 / 0 |
| sphinx-doc/sphinx | 20 | 5 | 7 | 3 / 1 | 7 | 4 / 2 |
