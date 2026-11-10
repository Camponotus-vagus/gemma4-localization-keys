# S3a repaired code graph (2026-10-24)

Reproduce: `uv run python -m experiments.s3_graph` (stages `build|analyze|localize|all`; needs S1 ground truth and S2 unit
indexes). Sanity tests: `uv run python tests/test_graph_builder.py`. Outputs: `results/s3_graph/{summary.json,per_task.jsonl}`,
graphs cached in `data/derived/graphs_repaired/`. 129 train tasks, 127 distinct graphs.

Builder: `src/g4a/graph_builder.py`, pure `ast` + `tarfile`, no execution, no third-party imports (runs in the agent
sandbox). Same JSON layout as the provided graphs (node-link, directed multigraph) and the same id convention
(dotted module path without `src/`, `docs_src/` + qualified name). Added: node attributes `path, lineno, end_lineno,
kind (function|async_function|class|module), qualname, is_test, legacy_id`; `module` nodes; edge types `contains`,
`calls` (import-aware: local and `from` imports with re-exports, relative imports, `self.m` and `super().m` through the MRO,
module attribute chains, constructors to `__init__`, light type inference from annotations and `x = Cls()`), `imports`,
`inherits`, `references` (names passed as call arguments, e.g. `Depends(get_db)`), `maybe_calls` (unresolved `obj.m()`
where `m` is a method defined once in the repo). Build time 2.4 s for FastAPI, 0.3 to 1.7 s for the others.

## Coverage of gold symbols (mirrors the S1 table)

| symbol kind | n | provided (S1, strict) | provided (lenient id match) | repaired |
|---|---|---|---|---|
| sync function / method | 381 | 377 (99%) | 381 | 381 |
| class | 63 | 61 (97%) | 63 | 63 |
| async function / method | 55 | 0 | 0 | 55 |
| module-level edit | 122 | 0 | 0 | 122 |
| new file | 49 | 0 | 0 | 0 (9 have a module of the same package as attach point) |

Tasks with at least one gold node: provided 115/129, repaired 128/129. Tasks with all gold symbols as nodes: provided
62/129 (65 lenient), repaired 118/129. The 11 remaining tasks create new files.

"Lenient": the provided builder names a function nested in a function without the outer name (`mod.Class.method`
instead of `mod.f.Class.method`), which explains the 4 sync and 2 class misses of S1; lenient matching accepts that
form. So the S1 gap is almost entirely async, module-level and new files.

## Graph composition (mean over 127 graphs)

| | provided | repaired |
|---|---|---|
| nodes | 2,898 | 4,218 (3,518 function/class + 700 module; 552 async) |
| edges | 3,564 (all `calls`) | 9,362 |
| calls / contains / imports | 3,564 / 0 / 0 | 4,126 / 3,518 / 1,184 |
| inherits / references / maybe_calls | 0 | 102 / 335 / 97 |
| test nodes | 57.4% | 53.5% of function/class nodes |
| isolated (no edge of any type) | 38.3% | 1.1% (only through `contains`) |
| isolated w.r.t. semantic edges (calls, inherits, references, maybe_calls), function/class only | 38.3% | 41.0% |

The 41% is not an improvement over the provided 38%: contains and imports connect everything, but semantic
isolation stays high because most nodes are tests and docs examples that nobody calls.

## Compatibility with the provided graphs

- Provided ids found among repaired `legacy_id`s: 98.9% (363,809 of 367,985). Exact for Rich, HTTPX and the current FastAPI
  graphs. The 4,176 misses: 1,887 in `requests`, where the provided builder loses the class prefix of every method after
  the first class defined inside a method of `tests/test_requests.py` (e.g. `tests.test_requests.test_certificate_failure`
  for `TestRequests.test_certificate_failure`; a second defect of the provided builder, missed by S1), and 2,289 in 52 older
  FastAPI commits, where docs examples carry a shorter module root (`app01.config.Settings` instead of
  `settings.app01.config.Settings`; 60 ids per graph). Not reproduced for those.
- Edge agreement (undirected pairs, nodes in both): 70% of provided pairs also have a semantic edge in the repaired graph
  (64% a `calls` edge). Of the repaired `calls` pairs among provided nodes, 52% exist in the provided graph. The
  remaining 30% of provided pairs look name-based (S1 already showed `Settings` <-> `get_settings` in both directions); not
  audited pair by pair.

## Localization: PPR seeded by BM25

Same gold, candidates (non-test functions, async functions, methods, classes from the S2 index), metric (hit rate at k)
and combination formula as S2: `0.5 * BM25/max + 0.5 * PPR/max`, 20 seeds, restart 0.25, 30 iterations. Repaired PPR runs on
the full graph (tests, modules included as nodes); only candidates are ranked. n = 128 tasks (file), 121 (unit).
`old_callgraph/mix0.5` reproduces the S2 `ppr` row exactly (0 mismatches over 4 k values x 121 tasks).

| method (20 seeds) | file@1 | file@5 | file@10 | unit@1 | unit@5 | unit@10 | unit@20 |
|---|---|---|---|---|---|---|---|
| BM25 over units | .336 | .727 | .820 | .157 | .380 | .529 | .603 |
| old call graph, mix .5 | .375 | .719 | .852 | .124 | .388 | .504 | .612 |
| repaired, all edges, mix .5 | .336 | .719 | .836 | .107 | .380 | .488 | .612 |
| repaired, no `contains` | .312 | .719 | .844 | .157 | .397 | .471 | .612 |
| repaired, no `calls` | .305 | .711 | .820 | .050 | .331 | .463 | .612 |
| repaired, no `imports` | .336 | .719 | .836 | .107 | .380 | .496 | .612 |
| repaired, no `inherits` | .336 | .719 | .836 | .107 | .388 | .479 | .612 |
| repaired, no `references` | .336 | .719 | .836 | .116 | .397 | .488 | .612 |
| repaired, no `maybe_calls` | .328 | .711 | .836 | .091 | .372 | .479 | .612 |
| repaired, `calls` only | .297 | .695 | .844 | .174 | .421 | .488 | .603 |
| repaired, calls+contains | .320 | .711 | .836 | .099 | .397 | .488 | .612 |
| repaired, semantic edges only | .320 | .719 | .844 | .165 | .380 | .479 | .612 |
| repaired, all edges, no test nodes | .336 | .719 | .844 | .091 | .388 | .479 | .603 |

Pure PPR ordering (no BM25 in the score) is worse than mix .5 in every variant (e.g. repaired all edges: unit@1 .099,
unit@10 .446, unit@20 .620); full table in `summary.json`.

Hybrid with leave-one-repo-out tuning of (seeds in {5,10,20,50}, mix in {.3,.5,.7,.85,.95}), objective mean of
unit@1/5/10 on the other three repos:

| method | file@1 | file@5 | file@10 | unit@1 | unit@5 | unit@10 | unit@20 |
|---|---|---|---|---|---|---|---|
| BM25 over units | .336 | .727 | .820 | .157 | .380 | .529 | .603 |
| hybrid, old call graph, LORO | .312 | .734 | .844 | .083 | .347 | .496 | .603 |
| hybrid, repaired graph, LORO | .328 | .719 | .820 | .149 | .372 | .537 | .620 |
| hybrid, repaired graph, best in-sample (50 seeds, mix .95; optimistic) | .328 | .719 | .820 | .149 | .388 | .545 | .620 |

Paired bootstrap (2,000 resamples over tasks) of LORO hybrid minus BM25: repaired graph unit@1 -.008 [-.025, 0.000],
unit@5 -.008 [-.025, 0.000], unit@10 +.008 [0.000, .025]; old graph unit@1 -.074 [-.132, -.025], unit@10 -.033 [-.083, .008].
LORO picks mix .95 in 3 of 4 folds, i.e. the tuner converges on "almost pure BM25".

Reachability from the BM25 top 20 (gold units outside the top 20; 74 of 121 tasks):

| graph | new candidates at 1 hop (mean) | tasks with a gold unit at 1 hop | new candidates at 2 hops | tasks with gold at 2 hops |
|---|---|---|---|---|
| old call graph | 112 | 43/74 | 326 | 59/74 |
| repaired, semantic edges | 61 | 34/74 | 320 | 56/74 |

## Interpretation

- Representability is fixed: async, module-level edits and (existing) files become nodes, 118/129 tasks have every gold
  symbol as a node versus 62/129. This is the solid result, and it is partly by construction (the gold map comes from the
  same AST view); the number to quote is the async 0/55 -> 55/55, which the provided builder cannot produce at all.
- Graph propagation does not improve retrieval over BM25 with either graph. The repaired graph is better than the old
  one in the tuned hybrid (unit@1 .149 against .083) only because the old graph's tuned hybrid is hurt more; against BM25
  there is no gain (differences within +-1 task in ~120, CIs touch 0).
- Ablations at 20 seeds are noisy (1 task = .008 in unit@k). The only consistent signal: removing `calls` hurts
  (unit@1 .050, unit@5 .331), so the call edges carry the useful information; `calls` alone is the best single-run
  variant at unit@1/5 (.174/.421) but this is a post-hoc pick among 13 variants and not tuned out-of-sample.
- Why it does not help: the graph does contain the gold unit within one or two hops of the BM25 top 20 for most tasks
  (34/74 and 56/74), but the neighbourhood holds 61 to 320 candidates, and PPR has no signal to choose among them. Neighbours
  are useful as candidates for an LLM or for a learned reranker, not as a scoring boost.

## Caveats

- Resolution is static: no dynamic dispatch, no `getattr`, inheritance only through bases resolvable in the repo,
  type inference limited to annotations and constructor assignments. No precision/recall of the individual edges was
  measured against a ground truth; evidence is the synthetic tests and the 52% / 70% agreement with the provided graph.
- `maybe_calls` and `references` are heuristic edge types; the `calls` type excludes them.
- Coverage numbers are on 4 public train repos (FastAPI 67, Rich 48, requests 13, HTTPX 1), the same ones used to design the
  builder; the hidden repos may expose resolution gaps. HTTPX is a single task, so LORO folds for it are trivially tuned
  on the other three.
- Candidates are the S2 units (no tests), so rankings stay comparable with S2; the graph contains test nodes which only act
  as intermediates in PPR.
- New files cannot be nodes by definition; 9 of 49 have a sibling module (attach point).
- n is small; differences under about 5 points are noise (bootstrap above).
