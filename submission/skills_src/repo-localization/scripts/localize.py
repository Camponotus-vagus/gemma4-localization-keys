"""BM25 localization over Python units (functions, methods, classes) of a repository.

Standard library only. Usage (from the skill runtime):
    localize.py --query "issue text or keywords" [--top 10] [--root /workspace]
    localize.py -- word1 word2 ...            (positional words are joined into the query)
"""

from __future__ import annotations

import argparse
import ast
import keyword
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

MAX_FILES = 6000
MAX_FILE_BYTES = 600_000
TIME_BUDGET_S = 20.0
MAX_OUTPUT_CHARS = 7000

SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv", "env", "build", "dist"}
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
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
        for p in [raw] + _CAMEL.sub(" ", raw).split():
            t = p.lower()
            if len(t) >= 2 and t not in STOP and not t.isdigit():
                out.append(t)
    return out


class Unit:
    __slots__ = ("path", "qualname", "kind", "start", "end", "text", "calls", "sig")

    def __init__(self, path, qualname, kind, start, end, text, calls, sig):
        self.path, self.qualname, self.kind = path, qualname, kind
        self.start, self.end, self.text, self.calls, self.sig = start, end, text, calls, sig


class _Calls(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name):
            self.names.add(f.id)
        elif isinstance(f, ast.Attribute):
            self.names.add(f.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:  # nested defs are their own units
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def _signature(lines: list[str], lineno: int) -> str:
    # Join a multi-line header (up to 4 lines) until the line that ends with a colon.
    parts: list[str] = []
    for ln in lines[lineno - 1:lineno + 3]:
        parts.append(ln.strip())
        if ln.rstrip().endswith(":"):
            break
    return " ".join(parts)


def units_in_file(path: str, source: str) -> list[Unit]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    lines = source.splitlines()
    out: list[Unit] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                kind = ("class" if isinstance(child, ast.ClassDef)
                        else "async def" if isinstance(child, ast.AsyncFunctionDef) else "def")
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = getattr(child, "end_lineno", None) or child.lineno
                col = _Calls()
                for stmt in child.body:
                    col.visit(stmt)
                sig = _signature(lines, child.lineno)
                out.append(Unit(path, qual, kind, start, end, "\n".join(lines[start - 1:end]),
                                sorted(col.names), sig))
                visit(child, qual)
            else:
                visit(child, prefix)  # defs nested in if/try blocks at module level

    visit(tree, "")
    return out


def collect(root: str, deadline: float) -> tuple[list[Unit], int, bool]:
    units: list[Unit] = []
    n_files = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
                             and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if TEST_PATH_RE.search(rel):
                continue
            if n_files >= MAX_FILES or time.monotonic() > deadline:
                truncated = True
                return units, n_files, truncated
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, "rb") as fh:
                    src = fh.read().decode("utf-8", errors="replace")
            except OSError:
                continue
            n_files += 1
            units.extend(units_in_file(rel, src))
    return units, n_files, truncated


def bm25_scores(docs: list[list[str]], query: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    n = len(docs)
    lens = [len(d) for d in docs]
    avgdl = (sum(lens) / n) if n else 1.0
    qf = Counter(query)
    postings: dict[str, list[tuple[int, int]]] = {t: [] for t in qf}
    for i, d in enumerate(docs):
        for t, c in Counter(d).items():
            if t in postings:
                postings[t].append((i, c))
    scores = [0.0] * n
    for t, q in qf.items():
        plist = postings[t]
        if not plist:
            continue
        df = len(plist)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        w = idf * (1 + math.log(q))
        for i, tf in plist:
            norm = k1 * (1 - b + b * lens[i] / avgdl)
            scores[i] += w * tf * (k1 + 1) / (tf + norm)
    return scores


def rank(units: list[Unit], query: str) -> list[int]:
    docs = [tokenize(u.path) * 2 + tokenize(u.qualname) * 3 + tokenize(u.text) for u in units]
    s = bm25_scores(docs, tokenize(query))
    return sorted(range(len(units)), key=lambda i: (-s[i], i))


def short(u: Unit) -> str:
    return f"{u.path}:{u.start}-{u.end}  {u.qualname}  ({u.kind})"


def render(units: list[Unit], order: list[int], top: int, n_files: int, truncated: bool) -> str:
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        by_name[u.qualname.rsplit(".", 1)[-1]].append(i)
    callers: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        for c in u.calls:
            callers[c].append(i)

    files: list[str] = []
    for i in order:
        p = units[i].path
        if p not in files:
            files.append(p)
        if len(files) >= top:
            break
    out = [f"Indexed {len(units)} units in {n_files} non-test .py files"
           + (" (truncated: repo too large)" if truncated else ""), "", f"Top {len(files)} files:"]
    out += [f"  {p}" for p in files]
    out += ["", f"Top {top} units (BM25; path:start-end  qualname  (kind)):"]
    for r, i in enumerate(order[:top], 1):
        u = units[i]
        out.append(f"{r:>2}. {short(u)}")
        if u.sig:
            out.append(f"      {u.sig[:110]}")
    out.append("")
    out.append("Neighbours of the top 3 (by simple-name match; may include homonyms):")
    for i in order[:3]:
        u = units[i]
        name = u.qualname.rsplit(".", 1)[-1]
        callees = []
        for c in u.calls:
            cands = by_name.get(c, [])
            if 0 < len(cands) <= 3:
                callees += [j for j in cands if j != i]
        cal = [j for j in callers.get(name, []) if j != i] if len(by_name[name]) <= 3 else []
        out.append(f"- {u.qualname} ({u.path})")
        out.append("    calls:     " + (", ".join(f"{units[j].qualname} [{units[j].path}:{units[j].start}]"
                                                 for j in dict.fromkeys(callees[:5])) or "-"))
        out.append("    called by: " + (", ".join(f"{units[j].qualname} [{units[j].path}:{units[j].start}]"
                                                 for j in dict.fromkeys(cal[:5])) or "-"))
    text = "\n".join(out)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    return text


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("words", nargs="*", help="query words (joined with spaces)")
    ap.add_argument("--query", default="", help="issue text or keywords")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--root", default="/workspace")
    ns = ap.parse_args(argv)
    query = " ".join([ns.query] + ns.words).strip()
    if not query:
        print("error: empty query; pass --query 'text' or positional words", file=sys.stderr)
        return 2
    root = ns.root if os.path.isdir(ns.root) else os.getcwd()
    units, n_files, truncated = collect(root, time.monotonic() + TIME_BUDGET_S)
    if not units:
        print(f"No Python units found under {root}.")
        return 0
    order = rank(units, query)
    print(render(units, order, max(1, min(ns.top, 40)), n_files, truncated))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
