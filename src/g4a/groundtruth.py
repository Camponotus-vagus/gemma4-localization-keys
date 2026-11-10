"""Localization ground truth from gold patches.

For each task, the gold patch tells which lines of the pre-fix repository were edited.
We map those lines to the innermost enclosing symbol (function, method, class) in the
pre-fix source, and then to a node id of the provided code graph when one exists.
"""

from __future__ import annotations

import ast
import json
import re
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


@dataclass
class FileEdit:
    path: str
    is_new_file: bool = False
    is_deleted_file: bool = False
    # Pre-image line numbers touched: removed/changed lines, plus the anchor line of pure insertions.
    old_lines: set[int] = field(default_factory=set)


@dataclass
class Symbol:
    path: str
    qualname: str  # "Class.method", "func", or "<module>"
    kind: str  # "function", "async_function", "class", "module"
    lineno: int
    end_lineno: int


@dataclass
class TaskTruth:
    instance_id: str
    repo: str
    base_commit: str
    files: list[str]
    test_like_files: list[str]
    new_files: list[str]
    symbols: list[dict]


def parse_patch(patch: str) -> list[FileEdit]:
    """Parse a unified diff (with or without `diff --git` headers) into touched pre-image lines.

    Hunk line counts are tracked so that a removed line whose content starts with "-- "
    is never mistaken for a new file header.
    """
    edits: list[FileEdit] = []
    cur: FileEdit | None = None
    old_path: str | None = None
    old_ln = old_left = new_left = 0

    def strip_prefix(p: str) -> str:
        p = p.split("\t", 1)[0].strip()
        return p[2:] if p[:2] in ("a/", "b/") else p

    for line in patch.splitlines():
        in_hunk = old_left > 0 or new_left > 0
        if in_hunk:
            if line.startswith("-"):
                cur.old_lines.add(old_ln)
                old_ln += 1
                old_left -= 1
            elif line.startswith("+"):
                # Pure insertion: anchor to the preceding pre-image line (at least 1).
                cur.old_lines.add(max(old_ln - 1, 1))
                new_left -= 1
            elif line.startswith(" ") or line == "":
                old_ln += 1
                old_left -= 1
                new_left -= 1
            continue  # "\ No newline at end of file" and anything else inside a hunk
        if line.startswith("--- "):
            old_path = line[4:]
        elif line.startswith("+++ ") and old_path is not None:
            new_path = line[4:]
            is_new = old_path.startswith("/dev/null")
            is_del = new_path.startswith("/dev/null")
            cur = FileEdit(path=strip_prefix(new_path if is_new else old_path),
                           is_new_file=is_new, is_deleted_file=is_del)
            edits.append(cur)
            old_path = None
        elif (m := HUNK_RE.match(line)) and cur is not None:
            old_ln = int(m.group(1))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
    return edits


def _symbols_in(source: str, path: str) -> list[Symbol]:
    tree = ast.parse(source)
    out: list[Symbol] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                kind = (
                    "class" if isinstance(child, ast.ClassDef)
                    else "async_function" if isinstance(child, ast.AsyncFunctionDef)
                    else "function"
                )
                start = min([child.lineno] + [d.lineno for d in getattr(child, "decorator_list", [])])
                out.append(Symbol(path, qual, kind, start, child.end_lineno or child.lineno))
                visit(child, qual)

    visit(tree, "")
    return out


def enclosing_symbols(source: str, path: str, lines: set[int]) -> list[Symbol]:
    """Innermost function/class containing each line; module-level lines map to '<module>'."""
    try:
        syms = _symbols_in(source, path)
    except SyntaxError:
        return [Symbol(path, "<module>", "module", 0, 0)]
    found: dict[str, Symbol] = {}
    for ln in sorted(lines):
        inner = [s for s in syms if s.lineno <= ln <= s.end_lineno]
        if inner:
            best = max(inner, key=lambda s: s.lineno)
        else:
            best = Symbol(path, "<module>", "module", 0, 0)
        found.setdefault(best.qualname, best)
    return list(found.values())


def module_candidates(path: str) -> list[str]:
    """Dotted module names for a file path, longest first (graphs drop some prefixes, e.g. docs_src)."""
    parts = path[:-3].split("/") if path.endswith(".py") else path.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]


def graph_node_id(sym: Symbol, node_ids: set[str]) -> str | None:
    if sym.kind == "module":
        return None
    for mod in module_candidates(sym.path):
        cand = f"{mod}.{sym.qualname}"
        if cand in node_ids:
            return cand
    return None


SnapshotSource = Path | IO[bytes]


def read_from_snapshot(snapshot: SnapshotSource, paths: list[str]) -> dict[str, str]:
    """Read selected files from a snapshot .tgz, given as a path or a readable byte stream."""
    wanted = {f"./{p}" for p in paths} | set(paths)
    out: dict[str, str] = {}
    opener = (
        tarfile.open(snapshot, "r:gz") if isinstance(snapshot, Path)
        else tarfile.open(fileobj=snapshot, mode="r|gz")
    )
    with opener as tf:
        for member in tf:
            if member.name in wanted and member.isfile():
                fh = tf.extractfile(member)
                if fh:
                    out[member.name.removeprefix("./")] = fh.read().decode("utf-8", errors="replace")
                if len(out) == len(paths):
                    break
    return out


def build_truth(task: dict, snapshot: SnapshotSource, graph_path: Path) -> TaskTruth:
    edits = parse_patch(task["patch"])
    py_edits = [e for e in edits if e.path.endswith(".py")]
    existing = [e.path for e in py_edits if not e.is_new_file]
    sources = read_from_snapshot(snapshot, existing) if existing else {}
    node_ids = {n["id"] for n in json.loads(graph_path.read_text())["nodes"]}

    symbols: list[dict] = []
    for e in py_edits:
        if e.is_new_file:
            symbols.append({**asdict(Symbol(e.path, "<new file>", "module", 0, 0)), "node_id": None, "in_graph": False})
            continue
        src = sources.get(e.path)
        if src is None:
            # Some gold patches create files without a /dev/null header: absent from the snapshot = new file.
            e.is_new_file = True
            symbols.append({**asdict(Symbol(e.path, "<new file>", "module", 0, 0)), "node_id": None, "in_graph": False})
            continue
        for s in enclosing_symbols(src, e.path, e.old_lines):
            nid = graph_node_id(s, node_ids)
            symbols.append({**asdict(s), "node_id": nid, "in_graph": nid is not None})

    return TaskTruth(
        instance_id=task["instance_id"],
        repo=task["repo"],
        base_commit=task["base_commit"],
        files=[e.path for e in edits],
        test_like_files=[e.path for e in edits if TEST_PATH_RE.search(e.path)],
        new_files=[e.path for e in edits if e.is_new_file],  # updated above for header-less new files
        symbols=symbols,
    )


def graph_path_for(task: dict, graphs_dir: Path) -> Path:
    short = task["repo"].split("/")[1].lower()
    return graphs_dir / f"{short}_{task['base_commit']}.json"
