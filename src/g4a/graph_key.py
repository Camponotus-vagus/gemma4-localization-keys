"""Graph-guided candidate sets for the identification key.

The repaired code graph is not used as a score: it expands the candidate set that the LLM key
chooses from. Seeds are the top-M BM25 units; their graph neighbours (calls, called-by, inherits,
references, contains parent/children) fill the remaining card slots. Only BM25 and the graph are
needed, both computable inside the agent sandbox.

Candidate builders return a `Cands`:
- `priority`: candidate uids in decreasing priority (seeds, then neighbours, then BM25 filler).
  It is the order used for retrieval-only metrics and as the tail of the final ranking.
- `display`: the order of the numbered cards in the prompt (equal to priority except for the
  grouped variant, where a neighbour is shown under its seed).
- `attach`: grouped variant only, neighbour uid -> (seed uid, relation label).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from g4a.key_localize import KeyResult, KeyLocalizer, SYSTEM, ISSUE_CHARS, _parse_list, _signature, unit_card
from g4a.repo_index import RepoIndex

SEMANTIC = ("calls", "inherits", "references", "maybe_calls")
# Relation label seen from the node a of an edge (a -> b) and from b.
_FWD = {"calls": "calls", "maybe_calls": "calls", "inherits": "related", "references": "related", "contains": "children"}
_BWD = {"calls": "called by", "maybe_calls": "called by", "inherits": "related", "references": "related",
        "contains": "parent"}


LABELS = ("calls", "called by", "parent", "children", "related")


@dataclass
class Cands:
    priority: list[str]
    display: list[str]
    attach: dict[str, tuple[str, str]] = field(default_factory=dict)


class GraphView:
    """Unit-level view of a repaired graph: relations between the units of a RepoIndex."""

    def __init__(self, idx: RepoIndex, graph: dict) -> None:
        nid = {n["id"]: i for i, n in enumerate(graph["nodes"])}
        self.is_test = [bool(n["is_test"]) for n in graph["nodes"]]
        self.is_module = [n["kind"] == "module" for n in graph["nodes"]]
        pq: dict[tuple[str, str], int] = {}
        for i, n in enumerate(graph["nodes"]):
            pq.setdefault((n["path"], n["qualname"]), i)
        self.uids = [u.uid for u in idx.units]
        self.unit_of_node: dict[int, int] = {}
        for ui, u in enumerate(idx.units):
            j = pq.get((u.path, u.qualname))
            if j is not None:
                self.unit_of_node.setdefault(j, ui)
        self.node_of_unit = {ui: j for j, ui in self.unit_of_node.items()}
        self.adj: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for e in graph["edges"]:
            t = e["type"]
            if t not in _FWD:
                continue
            a, b = nid[e["source"]], nid[e["target"]]
            if a != b:
                self.adj[a].append((b, _FWD[t]))
                self.adj[b].append((a, _BWD[t]))

    def expand(self, seeds: list[int], hops: int, via_tests: bool = False) -> dict[int, dict]:
        """Units within `hops` of the seed units (seeds excluded).

        Returns unit index -> {hop, links (distinct seeds reaching it within `hops`), best (rank of the
        best seed among `seeds`), label (relation to that seed at hop 1)}.
        """
        rank = {s: r for r, s in enumerate(seeds)}
        reach: dict[int, dict[int, set[int]]] = {}  # hop -> node -> seed set
        info: dict[int, dict] = {}
        seen_nodes: set[int] = set()
        frontier: dict[int, set[int]] = {}
        for s in seeds:
            j = self.node_of_unit.get(s)
            if j is not None:
                frontier.setdefault(j, set()).add(s)
                seen_nodes.add(j)
        for h in range(1, hops + 1):
            nxt: dict[int, set[int]] = {}
            labels: dict[int, tuple[int, str]] = {}
            for a, sset in frontier.items():
                for b, lab in self.adj.get(a, ()):
                    if b in seen_nodes and b not in nxt:
                        continue
                    nxt.setdefault(b, set()).update(sset)
                    if h == 1:
                        best = min(sset, key=rank.__getitem__)
                        if b not in labels or rank[best] < labels[b][0]:
                            labels[b] = (rank[best], lab)
            for b, sset in nxt.items():
                u = self.unit_of_node.get(b)
                if u is not None and u not in rank and u not in info:
                    info[u] = {"hop": h, "links": len(sset), "best": min(rank[s] for s in sset),
                               "label": labels.get(b, (0, "related"))[1]}
                elif u is not None and u in info and info[u]["hop"] == h:
                    info[u]["links"] = max(info[u]["links"], len(sset))
            seen_nodes |= set(nxt)
            # Intermediates: no modules (a module touches its whole file); tests only on request.
            frontier = {b: s for b, s in nxt.items() if not self.is_module[b] and (via_tests or not self.is_test[b])}
        return info


def bm25_cands(bm: np.ndarray, uids: list[str], k: int) -> Cands:
    order = [uids[i] for i in np.argsort(-bm, kind="stable")]
    return Cands(priority=order, display=order[:k])


def graph_cands(view: GraphView, bm: np.ndarray, k: int, m: int, hops: int, order: str = "bm25",
                grouped: bool = False, via_tests: bool = False) -> Cands:
    """Top-m BM25 seeds, then graph neighbours by `order`, then BM25 filler up to k cards.

    order "bm25": neighbours (all hops pooled) by BM25 score of the neighbour.
    order "links": nearer hops first, then by number of seeds linking to the neighbour, then BM25 score.
    order "samefile": nearer hops first, then neighbours in the file of one of the top-3 seeds, then BM25 score.
    """
    ranked = list(np.argsort(-bm, kind="stable"))
    seeds = [int(i) for i in ranked[:m]]
    info = view.expand(seeds, hops, via_tests)
    seed_files = {view.uids[s].split("::", 1)[0] for s in seeds[:3]}
    if order == "bm25":
        key = lambda u: (-bm[u], u)  # noqa: E731
    elif order == "samefile":
        key = lambda u: (info[u]["hop"], view.uids[u].split("::", 1)[0] not in seed_files, -bm[u], u)  # noqa: E731
    else:
        key = lambda u: (info[u]["hop"], -info[u]["links"], -bm[u], u)  # noqa: E731
    pool = [u for u in info if (not grouped or info[u]["hop"] == 1)]
    nbrs = sorted(pool, key=key)[: max(k - m, 0)]
    chosen = seeds + nbrs
    filler = [int(i) for i in ranked if int(i) not in set(chosen)][: max(k - len(chosen), 0)]
    chosen += filler
    uid = view.uids
    priority = [uid[i] for i in chosen] + [uid[int(i)] for i in ranked if int(i) not in set(chosen)]
    if not grouped:
        return Cands(priority=priority, display=priority[:k])
    attach = {uid[u]: (uid[seeds[info[u]["best"]]], info[u]["label"]) for u in nbrs}
    by_seed: dict[str, list[str]] = defaultdict(list)
    for u in sorted(nbrs, key=lambda u: LABELS.index(info[u]["label"])):  # stable: priority order inside a label
        by_seed[uid[seeds[info[u]["best"]]]].append(uid[u])
    display: list[str] = []
    for s in seeds:
        display.append(uid[s])
        display += by_seed.get(uid[s], [])
    display += [uid[i] for i in filler]
    return Cands(priority=priority, display=display, attach=attach)


# ---------------------------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------------------------

def variant_specs() -> dict[str, dict]:
    """name -> parameters. Names: bm25_k20 | g_m10_h2_links_k20 | gc_m10_k30 (c = grouped cards)."""
    specs: dict[str, dict] = {}
    for k in (20, 22, 25, 30, 40, 50):  # 22, 25, 40, 50: char-matched / ceiling references, not meant for LLM runs
        specs[f"bm25_k{k}"] = {"kind": "bm25", "k": k}
    for k in (20, 30):
        for m in (10, 15):
            for h in (1, 2):
                for order in ("bm25", "links", "samefile"):
                    specs[f"g_m{m}_h{h}_{order}_k{k}"] = {"kind": "graph", "k": k, "m": m, "hops": h, "order": order}
    for k in (20, 30, 40):  # grouped neighbours are one-line entries: k=40 costs about as many chars as flat k=30
        for m in (10, 15):
            specs[f"gc_m{m}_k{k}"] = {"kind": "grouped", "k": k, "m": m, "hops": 1, "order": "links"}
    # Diagnostic: 2-hop paths allowed to pass through test nodes (as in the S3a reachability count).
    specs["g_m10_h2_links_k30_tests"] = {"kind": "graph", "k": 30, "m": 10, "hops": 2, "order": "links", "via_tests": True}
    return specs


def build_cands(spec: dict, view: GraphView, bm: np.ndarray, uids: list[str]) -> Cands:
    if spec["kind"] == "bm25":
        return bm25_cands(bm, uids, spec["k"])
    return graph_cands(view, bm, spec["k"], spec["m"], spec["hops"], spec["order"],
                       grouped=spec["kind"] == "grouped", via_tests=spec.get("via_tests", False))


# ---------------------------------------------------------------------------------------------
# Grouped cards (variant c)
# ---------------------------------------------------------------------------------------------

def grouped_listing(idx: RepoIndex, c: Cands) -> str:
    """Seeds as full cards; each seed's neighbours as compact lines grouped by relation, all numbered."""
    by_uid = {u.uid: u for u in idx.units}
    num = {uid: i for i, uid in enumerate(c.display, 1)}
    kids: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for n, (seed, lab) in c.attach.items():
        kids[seed][lab].append(n)
    out: list[str] = []
    for uid in c.display:
        if uid in c.attach:
            continue
        out.append(f"[{num[uid]}] {unit_card(by_uid[uid])}")
        for lab in LABELS:
            ns = kids.get(uid, {}).get(lab, [])
            for n in ns:
                out.append(f"    {lab} -> [{num[n]}] {n}: {_signature(by_uid[n])[:100]}")
    return "\n".join(out)


def flat_key(loc: KeyLocalizer, idx: RepoIndex, issue: str, c: Cands, grouped: bool, k: int = 3) -> KeyResult:
    """The S2 flat key over `c.display`; the final ranking is picks, then c.priority (retrieval tail included)."""
    by_uid = {u.uid: u for u in idx.units}
    cands = c.display
    if grouped:
        listing = grouped_listing(idx, c)
        head = ("Candidate code units (some are shown indented under the unit they are related to; "
                "every numbered entry can be picked):\n")
    else:
        listing = "\n".join(f"[{i}] {unit_card(by_uid[u])}" for i, u in enumerate(cands, 1))
        head = "Candidate code units:\n"
    user = (
        f"Issue:\n{issue[:ISSUE_CHARS]}\n\n{head}{listing}\n\n"
        f"Which units must be edited to resolve the issue? Pick at most {k}, most likely first. "
        'Reply as {"choices": [numbers]}.'
    )
    res = KeyResult(ranking=[], picks=[])
    res.picks = _parse_list(loc._ask(user, res), cands)[:k]
    res.ranking = loc._complete(res.picks, c.priority)
    return res


def prompt_chars(idx: RepoIndex, c: Cands, grouped: bool) -> int:
    by_uid = {u.uid: u for u in idx.units}
    if grouped:
        return len(grouped_listing(idx, c))
    return sum(len(f"[{i}] {unit_card(by_uid[u])}\n") for i, u in enumerate(c.display, 1))
