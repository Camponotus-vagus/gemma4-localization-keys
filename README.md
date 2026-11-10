# Gemma 4 Developer Agent Competition: localization study

Code, experiments and writeup for a Kaggle Paper Track entry to the Gemma 4 Developer Agent Competition. The writeup is in `paper/draft_v0.md`. The numbers below come from that draft and may change in later versions.

The competition supplies per-commit code graphs and a `search_similar_code` tool for SWE agents. We audited both. The graphs omit every `async` function, use one edge type and leave 38% of nodes isolated. The embeddings are so anisotropic that the tool works as a name lookup. We release a stdlib-only AST graph builder that fixes these defects and a gold-patch localization benchmark (129 competition tasks, 153 SWE-bench Verified tasks from 8 repositories). Propagating scores over the call graph does not beat BM25. We then treat localization as an identification key, as in taxonomy. With Gemma 4 31B run locally, a flat multi-access key and a dichotomous key each raise unit@1 by about 0.3 over their own retrieval, on the training repositories and on held-out ones. On the training set the flat key beats the dichotomous key by 0.099, but this gap does not replicate on the held-out set (0.007). At equal accuracy the flat key needs one LLM call and about 40% fewer prompt tokens. The error classes of the key form a dichotomous key of their own, which we also release. End-to-end leaderboard results were still pending when the draft was written.

## Requirements

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/). The version is pinned in `.python-version` and `pyproject.toml`.
- Apple Silicon Mac. The key experiments use `mlx-community/gemma-4-31b-it-4bit` through `mlx-lm`. The runs were done on a 32 GB machine. The model and `Qwen/Qwen3-Embedding-0.6B` are downloaded from Hugging Face on first use.
- Competition data from Kaggle, placed in `data/competition/` (not included, see the data policy below). The folder holds the files the scripts read, including `tasks.jsonl`, `snapshots/` and `graphs/`.
- Four wheels in `data/wheelhouse/` (`swegemma`, `adk-submission`, `adk-eval-core`, `google-adk`), taken from the Kaggle dataset "Gemma 4 Developer Agent Wheelhouse". `uv sync` fails without them, because `pyproject.toml` points to these local paths.
- SWE-bench Verified test split from Hugging Face (`princeton-nlp/SWE-bench_Verified`) for the external set. `scripts/repro_runner.py` (`download_dataset()`) downloads the parquet file and writes `data/external/swebench_verified.jsonl`.
- Bare partial clones of the 8 external repositories, under `data/external/repos/` (commands in `paper/notes/s3_external.md`).

## Setup

```
source scripts/env.sh   # keeps uv, Hugging Face and pip caches and the Kaggle token inside the project
uv sync
```

Run every command from the project root. Quote paths in shell commands, because the folder name contains spaces.

## Reproducing the paper

Run the commands in the order of the table. Each script reads the outputs of the earlier steps listed in its notes under `paper/notes/`. `data/`, `results/` and `.cache/` are git-ignored. Each command writes its output there.

| Paper section | Command | Output |
|---|---|---|
| 3 Audit of the provided graph (S1) | `uv run python experiments/s1_groundtruth.py` then `uv run python experiments/s1_audit.py` | `data/derived/groundtruth.jsonl`, `results/s1_audit/` |
| 4.2 Localization baselines (S2) | `uv run python -m experiments.s2_localize` (after the S1 ground truth) | `results/s2_localize/` |
| 5 Identification keys, hybrid candidates (S2) | `uv run python -m experiments.s2_key retrieval`, then `run`, then `summary` | `results/s2_key/` |
| 5 Identification keys, BM25 candidates (S2) | `uv run python -m experiments.s2_key --variant bm25 retrieval`, then `run`, then `summary` | `results/s2_key_bm25/` |
| 4.1 and 4.3 Repaired graph (S3) | `uv run python -m experiments.s3_graph build`, then `analyze`, then `localize`. `all` runs the three stages | `results/s3_graph/`, `data/derived/graphs_repaired/` |
| 4.2 and 5 External set (S3b) | `uv run python -m experiments.s3_external <cmd>` for `select`, `snapshots`, `truth`, `baselines`, `baseline-summary`, `retrieval`, `key`, `summary`, in this order | `results/s3_external/`, `data/external/` |
| 4.3 Graph-expanded candidates (S3c) | `uv run python -m experiments.s3_graphkey --set train retrieval`, then `--set train summary`. Repeat with `--set external`. The `run` subcommand calls the LLM and is not used in the paper | `results/s3_graphkey/` and `results/s3_graphkey/external/` |
| 4.2 and 5 Statistics | `uv run --no-sync python -m experiments.s5_stats` | `results/s5_stats/summary.json` |
| 5.1 Error key (S5) | `uv run --no-sync python -m experiments.s5_error_key` | `results/s5_error_key/` |
| 6 Skill check (S4) | `uv run python -m experiments.s4_skill_check` | `results/s4_skill_check/` |
| Figures | `uv run python paper/figures/make_figures.py` | `paper/figures/fig_*.pdf` and `.png` |
| 4.1 Builder tests | `uv run python tests/test_graph_builder.py` (the file is pytest-compatible) | exit status only |
| 6 Submission bundles | `uv run python scripts/validate_submission.py submission/<bundle>` (no model or sandbox needed) | none |
| 6 Submission notebook | `uv run python scripts/build_submission_notebook.py submission/<bundle> --kaggle-user <user>` | `submission/build/kernels/` (git-ignored) |
| 6 Submission to Kaggle | `scripts/submit_bundle.sh submission/<bundle> "<description>"` | appends a row to `submission/LOG.md` |
| Public-data reproduction notebook | `uv run python scripts/build_repro_notebook.py --kaggle-user <user>`, then `uv run kaggle kernels push -p paper/notebook/g4a-repro` | `paper/notebook/g4a-repro/` |

Notes on the table:

- The `--variant` and `--set` options go before the subcommand name. Placed after it, argparse rejects them.
- `scripts/submit_bundle.sh` submits to the live competition, which allows one submission per day. It hard-codes the Kaggle user `francescosimone` in its second line, so change that before use.
- `scripts/build_submission_notebook.py` and `scripts/build_repro_notebook.py` take a Kaggle user name as an argument. `build_repro_notebook.py` defaults to `francescosimone`.
- The reproduction notebook is private. It uses only the SWE-bench Verified data. The Qwen3 embedding step is off by default, because it needs a GPU and takes a long time. Embedding scores can differ slightly by device and precision.
- The LLM commands (`s2_key run`, `s3_external key`) load the MLX model. They take a long time and are resumable.
- The paper's LLM experiments use the MLX 4-bit quantization, not the competition checkpoint, so absolute numbers are indicative.

## Repository layout

- `src/g4a/`: reusable code. `graph_builder.py` is the AST graph builder. `localize.py`, `embed_rank.py` and `key_localize.py` hold retrieval and the key prompts. `groundtruth.py` maps gold patches to functions. `graph_key.py`, `external.py`, `repo_index.py` and `wheels.py` cover the graph view, the external set, indexing and wheel handling.
- `experiments/`: one script per question, named after its study step (`s0` to `s5`). `s3_external_tasks.json` is the saved list of the 153 external tasks.
- `submission/`: agent bundles for the competition (`baseline_v0`, `v1_budget`, `v2_skill`), the `repo-localization` skill source in `skills_src/`, and `LOG.md` with submission hashes and scores.
- `paper/`: `draft_v0.md`, `figures/` (with `make_figures.py`), `notes/` (one note per study step, with the reproduction command), and `notebook/`.
- `tests/`: `test_graph_builder.py`, sanity checks on a synthetic repository.
- `scripts/`: environment setup (`env.sh`), submission tools, the notebook builders and the `repro_runner.py` driver.

## Data policy

Competition data and everything derived from it are not in this repository. This covers the task files, snapshots, graphs, embeddings, gold labels and result files that depend on the competition tasks. The organizers had not confirmed whether such derived data may be released. The scripts regenerate it from the competition files in `data/competition/`. The SWE-bench Verified tasks and the external task list (`experiments/s3_external_tasks.json`) come from public data.

## Licence

Apache License 2.0, see `LICENSE`.
