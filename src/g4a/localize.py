"""Localization baselines over a RepoIndex: BM25 (files, units) and personalized PageRank.

All rankers take the issue text and return an ordered list of unit ids ("path::qualname")
or file paths. No LLM is involved here.
"""

from __future__ import annotations

import keyword
import math
import re
from collections import Counter, defaultdict

import numpy as np

from g4a.repo_index import RepoIndex

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
STOP = set(keyword.kwlist) | {
    "self", "cls", "the", "and", "for", "that", "this", "with", "from", "not", "are", "was", "but",
    "have", "has", "can", "will", "would", "should", "when", "then", "than", "into", "use", "used",
    "none", "true", "false", "return", "def", "class", "import", "print", "str", "int", "dict", "list",
    "http", "https", "www", "com", "github",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in _SPLIT.split(text):
        if not raw:
            continue
        parts = [raw] + _CAMEL.sub(" ", raw).split()
        for p in parts:
            t = p.lower()
            if len(t) >= 2 and t not in STOP and not t.isdigit():
                out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.len.mean()) if len(docs) else 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        q = Counter(query)
        s = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avgdl)
        for term, qf in q.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.tf], dtype=float)
            s += idf * tf * (self.k1 + 1) / (tf + norm) * (1 + math.log(qf))
        return s


def unit_doc(u) -> list[str]:
    # Path and qualified name are repeated to weight identifiers over body text.
    return tokenize(u.path) * 2 + tokenize(u.qualname) * 3 + tokenize(u.text)


def rank_files_bm25(idx: RepoIndex, issue: str) -> list[str]:
    paths = sorted(idx.files)
    bm = BM25([tokenize(p) * 3 + tokenize(idx.files[p]) for p in paths])
    s = bm.scores(tokenize(issue))
    return [paths[i] for i in np.argsort(-s, kind="stable")]


def rank_units_bm25(idx: RepoIndex, issue: str) -> tuple[list[str], np.ndarray]:
    bm = BM25([unit_doc(u) for u in idx.units])
    s = bm.scores(tokenize(issue))
    order = np.argsort(-s, kind="stable")
    return [idx.units[i].uid for i in order], s


def call_graph(idx: RepoIndex) -> dict[int, set[int]]:
    """Undirected unit graph: caller <-> callee resolved by simple name, plus class <-> its methods."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(idx.units):
        by_name[u.qualname.rsplit(".", 1)[-1]].append(i)
    adj: dict[int, set[int]] = defaultdict(set)
    uid_to_i = {u.uid: i for i, u in enumerate(idx.units)}
    for i, u in enumerate(idx.units):
        for name in u.calls:
            targets = by_name.get(name, [])
            if 0 < len(targets) <= 5:  # skip very ambiguous names (e.g. "get", "run")
                for j in targets:
                    if j != i:
                        adj[i].add(j)
                        adj[j].add(i)
        if "." in u.qualname:
            parent = uid_to_i.get(f"{u.path}::{u.qualname.rsplit('.', 1)[0]}")
            if parent is not None:
                adj[i].add(parent)
                adj[parent].add(i)
    return adj


def personalized_pagerank(adj: dict[int, set[int]], n: int, seed: np.ndarray,
                          alpha: float = 0.25, iters: int = 30) -> np.ndarray:
    """alpha = restart probability. Dangling mass returns to the seed distribution."""
    p = seed / seed.sum() if seed.sum() > 0 else np.full(n, 1 / n)
    r = p.copy()
    deg = np.array([len(adj.get(i, ())) for i in range(n)], dtype=float)
    for _ in range(iters):
        nxt = np.zeros(n)
        dangling = r[deg == 0].sum()
        for i, nbrs in adj.items():
            if deg[i]:
                share = r[i] / deg[i]
                for j in nbrs:
                    nxt[j] += share
        r = alpha * p + (1 - alpha) * (nxt + dangling * p)
    return r


def rank_units_ppr(idx: RepoIndex, issue: str, top_seeds: int = 20,
                   mix: float = 0.5) -> list[str]:
    """Seed PPR with the top BM25 units, then mix normalized PPR and BM25 scores."""
    _, s = rank_units_bm25(idx, issue)
    n = len(idx.units)
    seed = np.zeros(n)
    top = np.argsort(-s)[:top_seeds]
    seed[top] = np.maximum(s[top], 0)
    r = personalized_pagerank(call_graph(idx), n, seed)
    bm = s / s.max() if s.max() > 0 else s
    pr = r / r.max() if r.max() > 0 else r
    combined = mix * bm + (1 - mix) * pr
    return [idx.units[i].uid for i in np.argsort(-combined, kind="stable")]


def files_from_units(unit_ranking: list[str]) -> list[str]:
    seen: list[str] = []
    for uid in unit_ranking:
        p = uid.split("::", 1)[0]
        if p not in seen:
            seen.append(p)
    return seen


def recall_at_k(ranking: list[str], gold: set[str], k: int) -> tuple[float, bool]:
    """(fraction of gold items in top-k, whether at least one is)."""
    if not gold:
        return float("nan"), False
    top = set(ranking[:k])
    hit = len(gold & top)
    return hit / len(gold), hit > 0
