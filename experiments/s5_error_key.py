"""S5: error taxonomy of the identification-key localizer, written as a dichotomous key.

Usage (after `source scripts/env.sh`): uv run python -m experiments.s5_error_key
Every miss at unit@1 (ranking[0] is not a gold unit) keys out to exactly one terminal class.
Characters used are observable in saved outputs only: the pick (ranking[0]), the model's raw reply,
the retrieval ranking, ground truth, and the repaired graph. One function per couplet, run in order.
Outputs: results/s5_error_key/per_miss.jsonl, counts.json, counts.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.s3_graphkey import Item, iter_items, load_jsonl
from g4a.graph_key import GraphView
from g4a.key_localize import _parse_list

OUT = Path("results/s5_error_key")
N_FLAT, N_FILES, K_FILES, N_UNITS = 20, 10, 2, 15  # key parameters of g4a.key_localize
GENERIC = {"get", "set", "is", "has", "to", "from", "make", "add", "create", "check", "init", "new", "run",
           "call", "do", "of", "in", "on", "the", "for", "with", "self", "cls", "main", "test"}

# terminal classes, in key order
CLASSES = ["A empty pick", "B1 ceiling: gold unit not in candidate pool", "B2 gold file not chosen (first couplet)",
           "B3 gold unit cut by the card limit", "C1 nesting (class/method of gold)",
           "C2 same-class sibling", "C3 same-file graph neighbour", "C4 same-file name-sharing unit",
           "C5 same-file other unit", "D1 cross-file graph neighbour", "D2 cross-file name-sharing unit",
           "D3 cross-file unrelated unit"]


class Case:
    """One miss, with everything the couplets may look at."""

    def __init__(self, it: Item, view: GraphView, mode: str, row: dict, retrieval: list[str]) -> None:
        self.it, self.view, self.mode, self.row, self.rk = it, view, mode, row, retrieval
        self.pick = row["ranking"][0]
        self.gold = it.gold_units
        self.gold_files = it.gold_files
        self.files: list[str] = []  # candidate files of the dichotomous key
        for uid in retrieval:
            p = uid.split("::", 1)[0]
            if p not in self.files:
                self.files.append(p)
            if len(self.files) == N_FILES:
                break
        if mode == "dichotomous":
            self.chosen = _parse_list(row["raw"][0], self.files)[:K_FILES] or self.files[:1]
            self.pool = [u for u in retrieval if u.split("::", 1)[0] in self.files]
            self.cards = [u for u in retrieval if u.split("::", 1)[0] in self.chosen][:N_UNITS]
        else:
            self.chosen = []
            self.pool = retrieval[:N_FLAT]
            self.cards = self.pool


def path_of(uid: str) -> str:
    return uid.split("::", 1)[0]


def qual_of(uid: str) -> str:
    return uid.split("::", 1)[1]


def tokens(uid: str) -> set[str]:
    leaf = qual_of(uid).split(".")[-1]
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", leaf).lower().split("_")
    return {p for p in parts if len(p) >= 3 and p not in GENERIC}


# ---- couplets: each returns True for the first lead, False for the second -------------------------

def c1_pick_empty(c: Case) -> bool:
    """1a. the model returned no parsable pick (ranking falls back to retrieval / chosen-file order)."""
    return not c.row["picks"]


def c2_gold_in_pool(c: Case) -> bool:
    """2a. a gold unit is in the candidate pool of the key (flat: top-20 units; dichotomous: units of the 10 files)."""
    return bool(set(c.pool) & c.gold)


def c3_gold_file_chosen(c: Case) -> bool:
    """3a. (dichotomous) a gold file is among the files chosen at the first couplet; flat key always passes."""
    return c.mode == "flat" or bool(set(c.chosen) & c.gold_files)


def c4_gold_on_cards(c: Case) -> bool:
    """4a. a gold unit is on the cards shown at the final choice."""
    return bool(set(c.cards) & c.gold)


def c5_pick_in_gold_file(c: Case) -> bool:
    """5a. the pick lies in a gold file."""
    return path_of(c.pick) in c.gold_files


def c6_nesting(c: Case) -> bool:
    """6a. pick and a gold unit are nested (class and its method, or method of a method)."""
    p = qual_of(c.pick)
    return any(path_of(g) == path_of(c.pick) and (qual_of(g).startswith(p + ".") or p.startswith(qual_of(g) + "."))
               for g in c.gold)


def c7_same_class(c: Case) -> bool:
    """7a. pick and a gold unit share the enclosing class/function (same parent qualname, different leaf)."""
    p = qual_of(c.pick)
    if "." not in p:
        return False
    return any(path_of(g) == path_of(c.pick) and "." in qual_of(g) and qual_of(g).rsplit(".", 1)[0] == p.rsplit(".", 1)[0]
               for g in c.gold)


def c8_graph_adjacent(c: Case) -> bool:
    """8a. pick is one hop from a gold unit in the repaired graph (calls, maybe_calls, inherits, references)."""
    v = c.view
    ui = {u: i for i, u in enumerate(v.uids)}
    pj = v.node_of_unit.get(ui.get(c.pick, -1))
    if pj is None:
        return False
    gj = {v.node_of_unit.get(ui[g]) for g in c.gold if g in ui} - {None}
    return any(b in gj and lab != "parent" and lab != "children" for b, lab in v.adj.get(pj, ()))


def c9_shared_token(c: Case) -> bool:
    """9a. the leaf name of the pick shares a (non-generic) word with the leaf name of a gold unit."""
    t = tokens(c.pick)
    return any(t & tokens(g) for g in c.gold)


def classify(c: Case) -> str:
    if c1_pick_empty(c):
        return CLASSES[0]
    if not c2_gold_in_pool(c):
        return CLASSES[1]
    if not c3_gold_file_chosen(c):
        return CLASSES[2]
    if not c4_gold_on_cards(c):
        return CLASSES[3]
    if c5_pick_in_gold_file(c):
        if c6_nesting(c):
            return CLASSES[4]
        if c7_same_class(c):
            return CLASSES[5]
        if c8_graph_adjacent(c):
            return CLASSES[6]
        return CLASSES[7] if c9_shared_token(c) else CLASSES[8]
    if c8_graph_adjacent(c):
        return CLASSES[9]
    return CLASSES[10] if c9_shared_token(c) else CLASSES[11]


def key_rows(dataset: str) -> dict[tuple[str, str], dict[str, dict]]:
    """(variant, mode) -> task -> key row."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    if dataset == "train":
        for v, d in (("hybrid", "results/s2_key"), ("bm25", "results/s2_key_bm25")):
            for r in load_jsonl(Path(d) / "keys.jsonl"):
                out[(v, r["mode"])][r["instance_id"]] = r
    else:
        for r in load_jsonl(Path("results/s3_external/keys.jsonl")):
            out[(r["variant"], r["mode"])][r["instance_id"]] = r
    return out


def retrievals(dataset: str) -> dict[str, dict[str, list[str]]]:
    if dataset == "train":
        paths = {"hybrid": "results/s2_key/retrieval.jsonl", "bm25": "results/s2_key_bm25/retrieval.jsonl"}
    else:
        paths = {v: f"results/s3_external/retrieval_{v}.jsonl" for v in ("hybrid", "bm25")}
    return {v: {r["instance_id"]: r["ranking"] for r in load_jsonl(Path(p))} for v, p in paths.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter] = {}
    n_task: Counter = Counter()
    multi: dict[str, Counter] = defaultdict(Counter)
    with (OUT / "per_miss.jsonl").open("w") as fh:
        for ds in ("train", "external"):
            keys, retr = key_rows(ds), retrievals(ds)
            for it in iter_items(ds):
                view = GraphView(it.idx, it.graph)
                for (v, mode), rows in keys.items():
                    row = rows.get(it.tid)
                    if row is None:
                        continue
                    cond = f"{ds}/{v}/{mode}"
                    n_task[cond] += 1
                    if row["ranking"][0] in it.gold_units:
                        continue
                    c = Case(it, view, mode, row, retr[v][it.tid])
                    cls = classify(c)
                    counts.setdefault(cond, Counter())[cls] += 1
                    multi[cond][cls] += len(it.gold_units) > 1
                    fh.write(json.dumps({
                        "condition": cond, "instance_id": it.tid, "class": cls, "pick": c.pick,
                        "n_gold_units": len(it.gold_units), "gold_units": sorted(it.gold_units),
                        "gold_files": sorted(it.gold_files), "chosen_files": c.chosen,
                        "gold_rank_in_retrieval": min((i for i, u in enumerate(c.rk) if u in it.gold_units), default=None),
                    }) + "\n")
    conds = sorted(n_task)
    (OUT / "counts.json").write_text(json.dumps(
        {c: {"n": n_task[c], "misses": sum(counts.get(c, {}).values()), "classes": dict(counts.get(c, {})),
             "multi_gold": dict(multi[c])} for c in conds}, indent=1) + "\n")
    head = "| class | " + " | ".join(c.replace("/", " ") for c in conds) + " |"
    lines = [head, "|" + "---|" * (len(conds) + 1),
             "| tasks | " + " | ".join(str(n_task[c]) for c in conds) + " |",
             "| misses @1 | " + " | ".join(str(sum(counts.get(c, {}).values())) for c in conds) + " |"]
    for k in CLASSES:
        lines.append(f"| {k} | " + " | ".join(str(counts.get(c, {}).get(k, 0)) for c in conds) + " |")
    (OUT / "counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
