# S4 repo-localization skill (2026-10-24)

Files: `submission/skills_src/repo-localization/{SKILL.md,scripts/localize.py}`; check: `uv run python -m experiments.s4_skill_check`
(outputs `results/s4_skill_check/`). Bundle using it: `submission/v2_skill`.

- Stdlib-only port of the S2 BM25-over-units ranking (same tokenization; path x2, qualname x3, body x1; k1 1.2, b 0.75).
  Indexes non-test `.py` under `/workspace` at call time (functions, async defs, methods, classes, line spans); skips hidden,
  build, dist, venv, site-packages; caps 6,000 files, 600 KB/file, 20 s, 7,000 output chars.
- Output: top files; top units as `path:start-end qualname (kind)` with signature; callers/callees by simple name for the top 3.
- Runtime contract (read from google/adk skill_toolset and swegemma agent_runner): `run_skill_script` needs no `load_skill`
  first; `args` dict becomes long options; script runs via runpy in a temp dir (cwd is not /workspace); stdout returns to the
  model; non-zero SystemExit returns no stdout; timeout = harness command timeout. The executor's `.adk_exec_*.py` in
  /workspace is excluded by `.git/info/exclude`, so it does not enter the patch. Sandbox image is python:3.13-slim.
- Skill name must be kebab-case and equal to its directory name (`repo_localization` fails validation).

Parity on 129 train tasks (query = problem statement, `--top 20`):

| | S2 BM25 units | skill |
|---|---|---|
| file@1 / @5 / @10 | .336 / .727 / .820 | .344 / .727 / .828 |
| unit@1 / @5 / @10 / @20 | .157 / .380 / .529 / .603 | .157 / .380 / .521 / .603 |

Differences come from 3 tasks (nested defs inside if/try indexed by the skill, skipped build/dist dirs). Runtime per task:
median 0.62 s, max 1.26 s. Output ~2k chars at `--top 10`.
