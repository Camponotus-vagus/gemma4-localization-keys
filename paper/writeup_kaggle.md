<!-- Kaggle Writeup fields, submitted 2026-11-11. Title (80-char limit): "Code Graphs Are Not Enough: Identification Keys for Localization with Gemma 4". Subtitle (140-char limit) as on Kaggle: "The provided graph tools miss every async function; an identification key lifts unit@1 by about .3; a BM25 skill leaves the score unchanged." Thumbnail: paper/figures/thumbnail_kaggle.png. Project links: GitHub repo and g4a-repro notebook. Body: everything after the subtitle line below. Regenerate from paper/draft_v0.md after any edit. -->
Subtitle: The provided graph tools miss every async function; an identification key lifts unit@1 by about .3 with Gemma 4 31B, while a BM25 skill leaves the leaderboard score unchanged.

## Abstract

The competition provides per-commit code graphs and a `search_similar_code` tool. We audited them: the graphs omit every `async` function (0 of 55 gold async symbols), contain one edge type and leave 38% of nodes isolated, and the embeddings are so anisotropic (participation ratio 2.4 of 256) that the tool reaches 10% recall of gold locations, against 20% for plain name lookup. We release a stdlib-only AST graph builder that fixes these defects and a gold-patch localization benchmark (129 competition tasks, 153 held-out SWE-bench Verified tasks). Propagating scores over the call graph does not beat BM25. We then frame localization as an identification key, borrowing from taxonomy. With Gemma 4 31B run locally, a flat multi-access key and a dichotomous key each lift unit@1 by about .3 over their own retrieval, on train and on held-out repositories. The flat key's train advantage over the dichotomous key (+.099, 95% CI [+.025, +.174]) does not replicate (+.007, [-.060, +.073]), but it needs one call and about 40% fewer prompt tokens. End to end, a simplified agent with a budget-oriented prompt scores about 0.15 on the public leaderboard (sample agent: 0.06, one submission). Adding a BM25 localization skill, which supplies the candidate list without the key step, changes the score by +0.004 (95% CI [-0.023, +0.031], six control and five skill submissions): no measurable effect.

## 1 Introduction

A dichotomous key in taxonomy asks one question at a time. Couplet 3 of a key to the world genera of winged ant queens (Cantone 2018), paraphrased:

> 3a. Antennae filiform: *Anochetus*.  
> 3b. Antennae weakly clavate or clavate: go to 4.

Antennal shape is a graded character, and one misread couplet sends the user down the wrong branch with no way back. Multi-access keys let the user consult characters in any order and tolerate a misread one. The author therefore converted Cantone's two keys (295 and 290 couplets) into a character matrix for a multi-access key to the 42 winged-ant genera of Italy.

Localization, deciding which file and function an issue requires editing, has the same structure: a wrong early choice wastes the remaining budget. The analogy gives two key designs to compare and metrics that map onto agent quantities: couplets per identification become LLM calls and tokens, and misidentification becomes editing the wrong place.

We expected repaired graphs to beat lexical retrieval and a hierarchical key to beat a flat one. The data refuted the first and did not support the second. Sections 3 to 6 cover the audit, the repaired builder and benchmark, the negative result on graph propagation, the keys with an error key, and an end-to-end A/B of our localization skill.

## 2 Setting

The model is fixed: `gemma-4-31b-it-qat-w4a16-ct`, 32k context, vLLM on 4 L4 GPUs, about 120 SWE-bench-like Python tasks from private repositories run sequentially within 12 hours. The public leaderboard uses 58 tasks, 0.017 each.

Experiments run on a MacBook Air M5 with `mlx-community/gemma-4-31b-it-4bit` (thinking off, greedy, max 200 tokens), a different quantization from the competition model, so absolute numbers are indicative.

The 129 training tasks cover four repositories (FastAPI 67, Rich 48, requests 13, HTTPX 1).

Ground truth comes from the gold patches: each pre-image line a patch touches maps (via `ast`) to its innermost enclosing function or class in the pre-fix source, matched to a graph node by dotted module suffix. Of the 129 tasks, 128 have a gold file and 121 a function or class unit. file@k and unit@k are hit rates (a gold item among the top k).

## 3 Auditing the provided code graph

Coverage of gold symbols (train, 129 tasks):

| symbol kind | n | in provided graph |
|---|---|---|
| sync function / method | 381 | 377 (99%) |
| class | 63 | 61 (97%) |
| async function / method | 55 | 0 (0%) |
| module-level edit | 122 | not representable |
| new file | 49 | not representable |

The builder drops every `async def`, so no graph has async nodes; the organizers confirmed that the hidden test graphs share this gap. Only 62 of 129 tasks have all gold symbols present as nodes.

Composition (mean over 127 graphs): about 2,900 nodes and 3,560 edges, all `calls`. Test code makes up 57% of nodes; 38% are isolated. Nodes carry no path or line number.

Embeddings (600-node sample per graph) are strongly anisotropic: random node pairs have mean cosine 0.83, the mean top-1 neighbour cosine is 0.997, and the participation ratio is 2.4 of 256 dimensions.

`search_similar_code` does not embed its query. It resolves the string to an existing node id (exact, dotted suffix, case-insensitive, then substring) and returns that node's precomputed neighbours, so natural-language queries work only as substrings of a node id.

We measured recall with queries extracted from the issue (backticked spans and identifiers, mean 4.1 per task) over the 115 tasks with a gold node. For 48 no query resolves to any node. A gold node appears in the union of top-10 results for 12 of 115 tasks (10%); name resolution alone reaches 23 (20%). The embedding step lowers recall: the queried node is excluded and its neighbours are near-random. In run logs shared on the competition forum, all 21 `search_similar_code` calls returned nothing.

## 4 A repaired graph and a localization benchmark

### 4.1 Builder

`src/g4a/graph_builder.py` uses only `ast` and `tarfile`, executes nothing, and runs inside the agent sandbox. It keeps the provided node-link layout and adds node attributes (`path`, `lineno`, `end_lineno`, `kind`, `qualname`, `is_test`), module nodes, and edge types `contains`, `calls`, `imports`, `inherits`, `references` (names passed as call arguments, e.g. `Depends(get_db)`) and `maybe_calls` (an unresolved `obj.m()` where `m` is defined once in the repo).

| | provided | repaired |
|---|---|---|
| async gold symbols as nodes | 0/55 | 55/55 |
| tasks with all gold symbols as nodes | 62/129 | 118/129 |
| edges (mean per graph) | 3,564, only `calls` | 9,362, six types |
| isolated nodes | 38.3% | 1.1% |

Isolation from semantic edges is still 41%, since most nodes are tests nothing calls. The 11 tasks still lacking nodes add new files. Coverage partly follows by construction, since the gold map uses the same AST view; only the async figure is independent.

### 4.2 Benchmark and baselines

Candidates are every function, method and class in non-test `.py` files; the query is the problem statement. Methods: BM25, Qwen3-Embedding-0.6B, reciprocal rank fusion (RRF) of the two, personalized PageRank (PPR) seeded by BM25 and mixed with it, and the provided tool. Train n is 121 to 128.

Results, train / external (the provided tool was not run externally):

| method (train / external) | file@1 | file@10 | unit@1 | unit@5 | unit@10 |
|---|---|---|---|---|---|
| BM25 over units | .336 / .477 | .820 / .876 | .157 / .199 | .380 / .450 | .529 / .550 |
| PPR, own call graph | .375 / .477 | .852 / .863 | .124 / .166 | .388 / .391 | .504 / .523 |
| Qwen3-Embedding-0.6B | .484 / .647 | .852 / .928 | .240 / .272 | .471 / .563 | .595 / .649 |
| RRF(embedding, BM25) | .422 / .634 | .914 / .974 | .165 / .305 | .446 / .603 | .636 / .689 |
| provided `search_similar_code` | .195 | .328 | .107 | .124 | .165 |

The external set holds 153 SWE-bench Verified tasks from 8 repositories (all of pytest, xarray, pylint and astropy; 20 each from Sphinx, scikit-learn, matplotlib and Django; seed 20261024).

Paired bootstrap (5,000 resamples throughout) gives PPR minus BM25 at unit@1 of -.033 on both sets ([-.099, +.033] train, [-.080, +.013] external): no gain. RRF beats BM25 at unit@10 on both sets (+.107 and +.139, intervals excluding 0). The provided tool is not separable from BM25 at unit@1 (-.050 [-.124, +.025]) and falls far behind at unit@10 (-.364 [-.463, -.265]).

### 4.3 The graph does not improve retrieval

PPR on the repaired graph (0.5 BM25 plus 0.5 PPR, max-normalized; 20 seeds; restart 0.25) with leave-one-repository-out tuning of seeds and mixing weight differs from BM25 by -.008 [-.025, 0.000] at unit@1 and +.008 [0.000, .025] at unit@10. The tuner picks almost pure BM25 in three of four folds. On the old graph the hybrid is worse (unit@1 -.074 [-.132, -.025]).

Reachability explains why. Of 74 tasks where the BM25 top 20 misses the gold unit, the graph reaches it within one hop in 34 and two in 56, but the neighbourhood holds 61 new candidates at one hop and about 320 at two.

As a candidate generator for an LLM (BM25 seeds plus one-hop neighbours, 30 cards), the best of 37 variants has a ceiling (share of tasks with a gold unit among the candidates) of .678 on train against .636 for BM25 and .711 for a BM25 list of 40; externally it ties BM25 (.695).

On these repositories, call-graph structure adds little over lexical and embedding retrieval. The repaired graph provides representability (async, module-level code, paths, line numbers) and the surroundings of a candidate.

## 5 Localization as an identification key

We give Gemma 4 31B a small, structured choice instead of an open search. The two keys share candidates and parser.

- Flat (multi-access): one call lists the top 20 units as cards and the model picks.
- Dichotomous: a first couplet chooses at most 2 of 10 candidate files, and a second picks among at most 15 units in them (two LLM calls).

The model's picks go first in the final ranking, followed by the retrieval order. Candidates come from the hybrid RRF or from BM25 alone, the only option in the agent sandbox. Train n is 121 and external n is 151.

| condition (train / external) | file@1 | unit@1 | unit@10 | calls | prompt tok |
|---|---|---|---|---|---|
| hybrid retrieval only | .438 / .636 | .165 / .305 | .636 / .689 | 0 | 0 |
| hybrid + flat | .785 / .861 | .579 / .623 | .694 / .768 | 1 | 1,241 / 1,997 |
| hybrid + dichotomous | .752 / .828 | .479 / .616 | .727 / .755 | 2 | 1,953 / 3,331 |
| BM25 only | .347 / .483 | .157 / .199 | .529 / .550 | 0 | 0 |
| BM25 + flat | .669 / .788 | .430 / .517 | .579 / .636 | 1 | 1,267 / 1,993 |
| BM25 + dichotomous | .702 / .755 | .446 / .510 | .603 / .669 | 2 | 1,965 / 3,278 |

On train, either key lifts unit@1 by .27 to .41 over retrieval alone. On hybrid candidates, flat minus dichotomous is +.099, 95% CI [+.025, +.174] (18 discordant pairs for, 6 against), and flat minus retrieval +.413 [+.314, +.504].

The dichotomous key picks a wrong file at its first couplet in 30 of 121 tasks, and in 11 of those the flat key still ranks a gold unit first: a wrong early couplet cannot be recovered, whereas the flat key lets later evidence rescue it. The flat key has a candidate ceiling instead: the gold unit is absent from its 20 cards in 30 of 121 tasks, capping unit@1 at .752, of which it reaches 77%. With BM25-only candidates the ordering reverses (.446 against .430), inside the noise: -.017 [-.099, +.058].

External replication (151 tasks, prompts unchanged; second number in each cell above). Either key again lifts unit@1 by about .3: flat minus retrieval is +.318 on both candidate sets (95% CI [+.225, +.411] hybrid, [+.238, +.397] BM25), and the flat key beats retrieval in each of the eight repositories. The flat-over-dichotomous advantage does not replicate: +.007 on hybrid candidates [-.060, +.073] and +.007 on BM25 candidates [-.053, +.073]. The flat key takes 22 s per task against 38.

The first-couplet failure recurs externally (25 of 151 tasks with hybrid candidates, none recovered by the dichotomous key). The flat ceiling is .801 (hybrid) and .642 (BM25).

On held-out data only cost separates the designs. The analogy has a limit: a taxonomic key uses characters chosen by an expert, while our cards are raw code.

### 5.1 Where the key fails

The error classes form a dichotomous key on characters observable in the logs (was a pick parsed, is a gold unit in the pool, is the pick in a gold file, are pick and gold nested or adjacent in the graph), so every miss keys out to exactly one class. It needs only saved outputs. We applied it to the 512 unit@1 misses across both keys, both candidate sets and both datasets.

| class | defining characters | misses | share |
|---|---|---|---|
| ceiling miss | no gold unit among the candidates | 214 | 42% |
| dichotomous-only structural | wrong file at couplet 1, or gold unit cut by the 15-card limit | 93 | 18% |
| empty pick | no parsable pick | 28 | 5% |
| near miss | pick nested with, in the same class as, or one graph hop from a gold unit | 123 | 24% |
| other wrong unit | any other pick with a gold unit on the cards | 54 | 11% |

Only retrieval fixes ceiling misses. Of the 177 choice errors (a gold unit on the cards, a different pick), 70% are near misses, so an agent that reads the picked code and its surroundings could recover. Shares are descriptive and the recovery is untested.

## 6 End-to-end agent

The official sample agent submitted unchanged (`baseline_v0`) scores 0.06.

`v1_budget` removes the sub-agent and the graph tools, and uses a prompt written around the budget (7 minutes, 30 calls): a first edit by call 10, and a clean diff even if a test is still red. `v2_skill` adds our `repo-localization` skill and one prompt paragraph that calls it. The skill is a stdlib-only port of the BM25 ranking of Section 4.2: it indexes the non-test Python files in `/workspace` when called and returns top files and units with line spans and signatures. Offline it matches the BM25 row (unit@10 .521 against .529).

One configuration resubmitted varies by about 3 tasks of 58 (SD about 1.6, from 10 resubmissions of a public bundle), so one submission cannot separate the arms. We alternated the two arms from 24 October to 10 November (scoring takes 24 to 33 hours) and compare them with a Mann-Whitney test and a bootstrap interval for the difference of means.

| arm | n | public scores | mean | SD |
|---|---|---|---|---|
| `baseline_v0` | 1 | 0.06 | | |
| `v1_budget` | 6 | 0.20, 0.18, 0.15, 0.13, 0.13, 0.12 | 0.152 | 0.032 |
| `v2_skill` | 5 | 0.18, 0.13, 0.15, 0.15, 0.17 | 0.156 | 0.019 |

`v2_skill` minus `v1_budget` is +0.004, 95% CI [-0.023, +0.031], about 0.2 task (Mann-Whitney U = 17.5, p = 0.75). The spread within each arm matches the resubmission noise. Dropping the first `v1_budget` score, submitted a day before the A/B protocol was fixed, gives +0.014. At this n a gain above about 2 tasks is unlikely; a smaller effect either way is not excluded.

The gain over the sample agent belongs to `v1_budget` as a whole (no sub-agent or graph tools, 7 minutes instead of 5, the budget prompt); we did not ablate it. The A/B tests retrieval only: the skill supplies a BM25 list (unit@1 .157 offline) and leaves the choice among cards, which lifted unit@1 by about .3 in Section 5, to the agent's own loop. The key itself is untested end to end, and localization may not be what limits an agent that resolves 15% of tasks. Task logs on the private set are not accessible, so we cannot tell how often the agent called the skill or whether it used the result.

## 7 Related work

SWE-bench (Jimenez 2024) and its human-validated subset Verified (OpenAI 2024) supply our held-out tasks. Agentless (Xia 2024) localizes hierarchically, from file to class or function to edit location; our dichotomous key is a two-level version. Graph-based localization for agents includes CodexGraph (Liu 2024), RepoGraph (Ouyang 2025), LocAgent (Chen 2025), OrcaLoca (Yu 2025) and CoSIL (Jiang 2025); these build their own graphs, while we audit the ones the competition provides. Our baselines are BM25 (Robertson and Zaragoza 2009), dense embeddings, reciprocal rank fusion (Cormack 2009) and personalized PageRank (Haveliwala 2002). Taxonomy contrasts multi-access keys such as DELTA and Intkey (Dallwitz 1980) with dichotomous keys (Walter and Winterton 2007). Ethayarajh (2019) describes embedding anisotropy.

## 8 Limitations

The train set has four repositories, two of which supply 89% of the tasks, and the test repositories are private. The external set is public and old, and Gemma may have seen it in pretraining; we ran no contamination test. Samples are small: one train task moves a rate by .008.

The LLM experiments use one quantization (MLX 4-bit), not the competition checkpoint, and prompts were not tuned. Valid edit sites other than the gold patch count as misses.

## 9 Reproducibility

All code is at https://github.com/Camponotus-vagus/gemma4-localization-keys (Apache-2.0); its README lists one command per experiment. The notebook https://www.kaggle.com/code/francescosimone/g4a-repro reruns the external benchmark from public data. Bundle hashes are in `submission/LOG.md`.

We release the graph builder, the benchmark scripts and the external task list (`experiments/s3_external_tasks.json`). Competition data stay on Kaggle; the organizers had not confirmed whether derived data may be released, so scripts regenerate labels.

## References

- Cantone S. (2018) Winged Ants, The Queen. Dichotomous key to genera of winged queen ants in the World. Stefano Cantone Editor, Catania.
- Chen Z. et al. (2025) LocAgent: graph-guided LLM agents for code localization. arXiv:2503.09089.
- Cormack G.V., Clarke C.L.A., Büttcher S. (2009) Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Dallwitz M.J. (1980) A general system for coding taxonomic descriptions. Taxon 29.
- Ethayarajh K. (2019) How contextual are contextualized word representations? EMNLP-IJCNLP.
- Haveliwala T.H. (2002) Topic-sensitive PageRank. WWW.
- Jiang Z. et al. (2025) Issue localization via LLM-driven iterative code graph searching (CoSIL). arXiv:2503.22424.
- Jimenez C.E. et al. (2024) SWE-bench: can language models resolve real-world GitHub issues? ICLR.
- Liu X. et al. (2024) CodexGraph: bridging large language models and code repositories via code graph databases. arXiv:2408.03910.
- OpenAI (2024) Introducing SWE-bench Verified. openai.com.
- Ouyang S. et al. (2025) RepoGraph: enhancing AI software engineering with repository-level code graph. ICLR.
- Robertson S., Zaragoza H. (2009) The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval 3(4).
- Walter D.E., Winterton S. (2007) Keys and the crisis in taxonomy: extinction or reinvention? Annual Review of Entomology 52.
- Xia C.S. et al. (2024) Agentless: demystifying LLM-based software engineering agents. arXiv:2407.01489.
- Yu Z. et al. (2025) OrcaLoca: an LLM agent framework for software issue localization. ICML.
