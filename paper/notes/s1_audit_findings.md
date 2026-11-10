# S1 audit findings (2026-09-28)

Reproduce: `uv run python experiments/s1_groundtruth.py && uv run python experiments/s1_audit.py`
(outputs in `results/s1_audit/`). Train split only: 129 tasks, 127 distinct graphs.

## Ground truth

Gold patch → touched pre-image lines → innermost enclosing def/class in the pre-fix source (AST) →
graph node id by matching dotted module suffixes (graphs drop prefixes such as `docs_src/`).
Pure insertions are anchored to the preceding pre-image line. Files missing from the snapshot are
treated as new files (some gold patches create files without a `/dev/null` header).

## Coverage of gold symbols by graph nodes

| Symbol kind | n | in graph |
|---|---|---|
| sync function / method | 381 | 377 (99%) |
| class | 63 | 61 (97%) |
| async function / method | 55 | 0 (0%) |
| module-level edit | 122 | 0 (not representable) |
| new file | 49 | 0 (not representable) |

Tasks with at least one gold node: 115/129. Tasks with all gold symbols as nodes: 62/129 (48%).
The graph builder drops every `async def` (0 async nodes in 127 graphs), including in FastAPI,
where request handlers and docs examples are mostly async.

## Graph composition (mean over 127 graphs)

- ~2,900 nodes, ~3,560 edges; the only edge type is `calls`.
- 57% of nodes are tests; 38% of nodes are isolated (degree 0).
- Nodes carry id, name and source text only: no file path, no line numbers.

## Embedding geometry (600-node sample per graph, mean over graphs)

- Mean pairwise cosine between random nodes: 0.83 (5th percentile 0.51).
- Mean top-1 neighbour cosine: 0.997.
- PC1 explains 63% of variance; participation ratio 2.4 out of 256 dimensions.
- Mean resultant length of unit vectors: 0.91 (1.0 = all parallel).

Interpretation: strongly anisotropic; similarity rankings are dominated by a shared direction, so
cosine scores carry little discriminative information unless the space is re-centred/whitened.

## How `search_similar_code` actually works

It does not embed the query text. `get_similar_nodes` resolves the query string to an existing
node id (exact, dotted-suffix, case-insensitive, then substring match) and returns the nearest
neighbours of that node's precomputed embedding, excluding the node itself. A natural-language
query only works if it happens to be a substring of some node id. The tool description given to
the agent ("find semantically similar functions/classes by keyword") does not say this.

## Tool recall with issue-derived queries

Queries: backticked spans and code-like identifiers from the problem statement (≤12 per task,
mean 4.1). 115 tasks with ≥1 gold node.

- Queries that resolve to any node: 33%. Tasks where no query resolves: 48/115.
- Gold node among the union of top-10 results: 12/115 (10%).
- Gold node reached by name resolution alone (the lookup step, before embeddings): 23/115 (20%).

The embedding step lowers recall relative to plain name lookup, because the queried node itself is
excluded and its neighbours are near-random under this anisotropy.

## Caveats

- Query extraction is a heuristic; a model may issue better queries. Next: compare against queries
  actually emitted by Gemma in agent traces.
- Train repos only (4 public repos, heavily skewed: 67 fastapi, 48 rich); the hidden test set comes
  from private repos built "with identical graph generation methods" (data page), so the async gap
  and the tool semantics should transfer; the magnitudes may not.
