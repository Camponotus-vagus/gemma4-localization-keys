# S5 error key for the identification-key localizer (2026-10-24)

Reproduce: `uv run --no-sync python -m experiments.s5_error_key` -> `results/s5_error_key/{per_miss.jsonl,counts.json,counts.md}`.
Scope: every miss at unit@1 (ranking[0] is not a gold unit) of the flat and dichotomous keys, hybrid and BM25 candidates,
121 train tasks (S2) and 151 external tasks (S3b): 8 conditions, 512 misses. Each keys out to exactly one terminal class.
Characters are read from saved outputs only: the model's reply, the retrieval ranking, ground truth, the repaired graph.
"Pool" = candidates the key can reach (flat: top-20 units; dichotomous: retrieved units of the 10 candidate files).
"Cards" = units shown at the final choice (flat: top-20; dichotomous: first 15 retrieved units of the chosen files).
"Adjacent" = one hop in the repaired graph by calls, maybe_calls, inherits or references (contains is tested separately).
"Shares a name" = leaf names share a word after splitting snake/camel case, words under 3 letters and generic verbs
(get, set, is, make, add, init, ...) removed.

## Key

1a. Reply gave no parsable pick (empty `picks`) .......... **A** empty pick
1b. A pick exists ........................................ 2
2a. No gold unit in the pool ............................. **B1** ceiling miss
2b. A gold unit is in the pool ........................... 3
3a. Flat key, or a gold file is among the files chosen ... 4
3b. Dichotomous key and no gold file chosen at couplet 1 . **B2** first-couplet error
4a. A gold unit is on the cards ........................... 5
4b. Gold unit in the chosen file but cut by the 15-card limit .. **B3** card-limit miss
5a. The pick lies in a gold file .......................... 6
5b. The pick lies in another file ......................... 9
6a. Pick and a gold unit are nested (class and its method) .. **C1** nesting
6b. Not nested ............................................ 7
7a. Pick and a gold unit have the same parent qualname ..... **C2** same-class sibling
7b. Different parent ...................................... 8
8a. Pick adjacent to a gold unit in the graph ............. **C3** same-file graph neighbour
8b. Not adjacent: shares a name with a gold unit .......... **C4** same-file name-sharing unit
    Not adjacent, no shared name ........................... **C5** same-file other unit
9a. Pick adjacent to a gold unit in the graph ............. **D1** cross-file graph neighbour
9b. Not adjacent: shares a name with a gold unit .......... **D2** cross-file name-sharing unit
    Not adjacent, no shared name ........................... **D3** cross-file unrelated unit

Couplet order matters: A is tested first, so an empty pick whose gold is also outside the pool counts as A.
Multi-location gold (more than one gold unit) is a flag in `per_miss.jsonl`, not a couplet: it applies across classes.

## Counts

| class | train hybrid flat | train hybrid dich | train bm25 flat | train bm25 dich | ext hybrid flat | ext hybrid dich | ext bm25 flat | ext bm25 dich | all |
|---|---|---|---|---|---|---|---|---|---|
| tasks | 121 | 121 | 121 | 121 | 151 | 151 | 151 | 151 | |
| misses @1 | 51 | 63 | 69 | 67 | 57 | 58 | 73 | 74 | 512 |
| A empty pick | 0 | 3 | 8 | 4 | 0 | 1 | 1 | 11 | 28 |
| B1 ceiling miss | 30 | 9 | 41 | 21 | 30 | 10 | 53 | 20 | 214 |
| B2 first-couplet error | 0 | 18 | 0 | 13 | 0 | 18 | 0 | 15 | 64 |
| B3 card-limit miss | 0 | 7 | 0 | 10 | 0 | 6 | 0 | 6 | 29 |
| C1 nesting | 2 | 5 | 4 | 4 | 8 | 6 | 4 | 7 | 40 |
| C2 same-class sibling | 6 | 9 | 6 | 8 | 2 | 3 | 4 | 5 | 43 |
| C3 same-file neighbour | 3 | 8 | 2 | 6 | 4 | 5 | 1 | 5 | 34 |
| C4 same-file name-sharing | 1 | 2 | 0 | 0 | 2 | 4 | 1 | 2 | 12 |
| C5 same-file other | 2 | 1 | 2 | 0 | 4 | 5 | 1 | 2 | 17 |
| D1 cross-file neighbour | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 0 | 6 |
| D2 cross-file name-sharing | 4 | 0 | 3 | 0 | 1 | 0 | 1 | 0 | 9 |
| D3 cross-file unrelated | 3 | 1 | 3 | 1 | 2 | 0 | 5 | 1 | 16 |

Of the 177 choice errors (C and D), 86 (49%) have more than one gold unit (C1 31, C2 18, C4 11, C3 8, D3 6, C5 5, D2 4, D1 3).
B1 misses: for 83 of 214 the gold unit is outside the top 200 retrieved; the others have median retrieval rank 59.

## Reading

Pooled over conditions, 42% of misses are ceiling misses (B1) that only better retrieval can fix, 18% are structural
losses of the dichotomous key (B2 first-couplet error, B3 card limit; 0 by construction for the flat key), 5% are empty
picks, and 35% are choice errors with the gold unit on the cards. Of those 177 choice errors, 123 (70%) are near misses
(nesting, same-class sibling, graph neighbour): the pick sits in the right file or one graph hop from a gold unit, and an
agent that reads the surrounding code should recover. In the hybrid flat key, where retrieval is best and the
structural classes vanish, ceiling misses are 59% (train) and 53% (external) of the remaining errors. The dichotomous key
converts ceiling misses into B2/B3 errors: its B1 count is lower (9-10 against 30 on hybrid) but B2+B3 adds 24-25.
Class counts are small (n < 20 for most cells per condition); the pooled shares are descriptive, not tested.
