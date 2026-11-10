"""Symbol index built directly from a repository snapshot (independent of the provided graphs).

Every function, async function, method and class in non-test Python files becomes a unit
with its path, qualified name, line span, source text and the simple names it calls.
"""

from __future__ import annotations

import ast
import gzip
import json
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

from g4a.groundtruth import TEST_PATH_RE


@dataclass
class Unit:
    path: str
    qualname: str
    kind: str  # function | async_function | class
    lineno: int
    end_lineno: int
    text: str
    calls: list[str] = field(default_factory=list)  # simple callee names (last dotted component)

    @property
    def uid(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass
class RepoIndex:
    files: dict[str, str]  # path -> source (non-test .py only)
    units: list[Unit]

    def to_json(self) -> dict:
        return {"files": self.files, "units": [asdict(u) for u in self.units]}

    @classmethod
    def from_json(cls, d: dict) -> RepoIndex:
        return cls(files=d["files"], units=[Unit(**u) for u in d["units"]])


def is_candidate_file(path: str) -> bool:
    return path.endswith(".py") and not TEST_PATH_RE.search(path) and "/.git/" not in f"/{path}"


def read_python_files(snapshot: Path | IO[bytes]) -> dict[str, str]:
    opener = (
        tarfile.open(snapshot, "r:gz") if isinstance(snapshot, Path)
        else tarfile.open(fileobj=snapshot, mode="r|gz")
    )
    out: dict[str, str] = {}
    with opener as tf:
        for m in tf:
            name = m.name.removeprefix("./")
            if m.isfile() and is_candidate_file(name):
                fh = tf.extractfile(m)
                if fh:
                    out[name] = fh.read().decode("utf-8", errors="replace")
    return out


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name):
            self.names.append(f.id)
        elif isinstance(f, ast.Attribute):
            self.names.append(f.attr)
        self.generic_visit(node)

    # Do not descend into nested defs: they are their own units.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass


def units_in_file(path: str, source: str) -> list[Unit]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: list[Unit] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                kind = (
                    "class" if isinstance(child, ast.ClassDef)
                    else "async_function" if isinstance(child, ast.AsyncFunctionDef)
                    else "function"
                )
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                col = _CallCollector()
                for stmt in child.body:
                    col.visit(stmt)
                out.append(Unit(path, qual, kind, start, end, "\n".join(lines[start - 1:end]), sorted(set(col.names))))
                visit(child, qual)

    visit(tree, "")
    return out


def build_index(snapshot: Path | IO[bytes]) -> RepoIndex:
    files = read_python_files(snapshot)
    units = [u for path, src in sorted(files.items()) for u in units_in_file(path, src)]
    return RepoIndex(files=files, units=units)


def cached_index(snapshot_path: Path, cache_dir: Path, key: str) -> RepoIndex:
    cache = cache_dir / f"{key}.json.gz"
    if cache.exists():
        return RepoIndex.from_json(json.loads(gzip.decompress(cache.read_bytes())))
    idx = build_index(snapshot_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(idx.to_json()).encode()))
    return idx
