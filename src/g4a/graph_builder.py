"""Repaired code-graph builder (pure AST, standard library only).

Builds a graph from Python sources without executing anything. The JSON layout follows the
provided graphs (networkx node-link: directed multigraph, nodes with id/name/text, edges with
type/source/target/key) so that the same ids can be mapped, and adds what they lack:

  nodes  function | async_function | class | module, with path, lineno, end_lineno, qualname, is_test
  edges  contains   module -> class/function, class -> method, function -> nested def
         calls      import-aware resolution (local/from imports, self.method through the MRO,
                    module attribute chains, simple local/attribute type inference, constructors)
         imports    module -> module
         inherits   class -> base class (when the base resolves inside the repo)
         references names or attributes passed as call arguments (e.g. `Depends(get_db)`)
         maybe_calls unresolved `obj.name()` where `name` is a method defined exactly once in the repo

Node ids follow the provided convention: dotted module path (leading `src`/`docs_src` dropped)
plus the qualified name, e.g. `fastapi.routing.APIRoute.get_route_handler`. Module nodes use the
dotted module name. Repeated definitions of one qualified name (overloads, property setters) get
a `#2`, `#3` suffix. Only `ast` and `tarfile` are used so the module can run in an agent sandbox.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, Iterator

ROOT_PREFIXES = ("docs_src", "src")
SKIP_SEGMENTS = {".git", ".tox", ".venv", "venv", "node_modules", "site-packages", "__pycache__"}
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
MAX_TEXT = 3000
EDGE_TYPES = ("contains", "calls", "imports", "inherits", "references", "maybe_calls")
_DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)

# A resolution target: ("mod", module_name) | ("node", node_id) | ("inst", class_id) | ("super", class_id) | ("ext",)
Target = tuple


def read_python_files(snapshot: Path | IO[bytes]) -> dict[str, str]:
    """All .py files of a snapshot tarball (tests included), keyed by repo-relative path."""
    opener = (
        tarfile.open(snapshot, "r:gz") if isinstance(snapshot, Path)
        else tarfile.open(fileobj=snapshot, mode="r|gz")
    )
    out: dict[str, str] = {}
    with opener as tf:
        for m in tf:
            name = m.name.removeprefix("./")
            if m.isfile() and name.endswith(".py") and not SKIP_SEGMENTS & set(name.split("/")):
                fh = tf.extractfile(m)
                if fh:
                    out[name] = fh.read().decode("utf-8", errors="replace")
    return out


def module_name_of(path: str) -> tuple[str, bool]:
    """(dotted module name, is_package) for a .py path; `src/` and `docs_src/` roots are dropped."""
    parts = path[:-3].split("/")
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] in ROOT_PREFIXES:
        parts = parts[1:]
    return (".".join(parts) if parts else "__root__"), is_pkg


@dataclass
class _Mod:
    name: str
    path: str | None  # None for virtual namespace packages
    is_pkg: bool
    tree: ast.Module | None = None
    defs: dict[str, str] = field(default_factory=dict)  # top-level name -> node id
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # local -> (module, symbol)
    var_types: dict[str, str] = field(default_factory=dict)  # module-level variable -> class id
    vt_state: int = 0  # 0 not computed, 1 running, 2 done
    node_id: str | None = None


@dataclass
class _Node:
    id: str
    qualname: str
    kind: str
    path: str
    lineno: int
    end_lineno: int
    text: str
    is_test: bool
    mod: str
    parent: str | None
    ast: ast.AST | None
    cls: str | None = None  # enclosing class when this def is a direct class member
    legacy_id: str = ""  # id in the provided-graph convention: enclosing function names are dropped
    members: dict[str, str] = field(default_factory=dict)  # class members / function-local defs


@dataclass
class _Ctx:
    mod: _Mod
    node_id: str
    owner_cls: str | None
    local_defs: dict[str, str]
    imports: dict[str, tuple[str, str | None]]
    var_types: dict[str, str]
    bound: set[str]


def _walk_scope(body: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Nodes of one scope. Nested defs are yielded, but only their decorators, defaults and bases are entered."""
    stack = list(reversed(list(body)))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, _FUNC):
            a = n.args
            stack.extend(n.decorator_list)
            stack.extend(a.defaults + [d for d in a.kw_defaults if d is not None])
        elif isinstance(n, ast.ClassDef):
            stack.extend(n.decorator_list + n.bases + [k.value for k in n.keywords])
        else:
            stack.extend(ast.iter_child_nodes(n))


def _sub_blocks(st: ast.AST) -> list[list[ast.stmt]]:
    out: list[list[ast.stmt]] = []
    for name in ("body", "orelse", "finalbody"):
        v = getattr(st, name, None)
        if isinstance(v, list):
            out.append(v)
    out.extend(h.body for h in getattr(st, "handlers", []))
    out.extend(c.body for c in getattr(st, "cases", []))
    return out


def _iter_defs(stmts: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Definitions reachable without entering another def (so also under if/try/with/for)."""
    for st in stmts:
        if isinstance(st, _DEF):
            yield st
        else:
            for block in _sub_blocks(st):
                yield from _iter_defs(block)


def _dotted(expr: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


class _Builder:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.mods: dict[str, _Mod] = {}
        self.nodes: dict[str, _Node] = {}
        self.order: list[str] = []
        self.edges: dict[tuple[str, str, str], None] = {}
        self.ctxs: dict[str, _Ctx] = {}
        self.ctx_running: set[str] = set()
        self.bases: dict[str, list[str]] = {}
        self.attr_types: dict[str, dict[str, str]] = {}
        self.attr_running: set[str] = set()
        self._suffix: dict[str, list[str]] | None = None
        self._method_index: dict[str, list[str]] | None = None
        self.bases_ready = False

    # ---- pass 1: modules and definitions -------------------------------------------------

    def run(self) -> None:
        for path in sorted(self.files):
            name, is_pkg = module_name_of(path)
            if name in self.mods and self.mods[name].path is not None:
                continue  # a/b.py and a/b/__init__.py: keep the first
            try:
                tree = ast.parse(self.files[path])
            except (SyntaxError, ValueError, RecursionError):
                tree = None
            self.mods[name] = _Mod(name, path, is_pkg, tree)
        for name in list(self.mods):
            parts = name.split(".")
            for i in range(1, len(parts)):
                self.mods.setdefault(".".join(parts[:i]), _Mod(".".join(parts[:i]), None, True))
        for mod in list(self.mods.values()):
            if mod.path is not None:
                self._register_module(mod)
        for mod in self.mods.values():
            if mod.tree is not None:
                self._collect_imports(mod)
        for nid in list(self.order):
            self._resolve_bases(nid)
        self.bases_ready = True
        for nid in list(self.order):
            self._emit_scope_edges(nid)

    def _unique_id(self, base: str) -> str:
        if base not in self.nodes:
            return base
        i = 2
        while f"{base}#{i}" in self.nodes:
            i += 1
        return f"{base}#{i}"

    def _add_node(self, **kw) -> _Node:
        n = _Node(**kw)
        self.nodes[n.id] = n
        self.order.append(n.id)
        return n

    def _register_module(self, mod: _Mod) -> None:
        assert mod.path is not None
        src = self.files[mod.path]
        n_lines = src.count("\n") + 1
        doc = ast.get_docstring(mod.tree) if mod.tree is not None else None
        text = (f'"""{doc}"""\n' if doc else "") + "\n".join(
            ln for ln in src.splitlines()[:80] if ln.startswith(("import ", "from ")))
        self._add_node(id=self._unique_id(mod.name), qualname="<module>", kind="module", path=mod.path,
                       lineno=1, end_lineno=n_lines, text=text[:MAX_TEXT],
                       is_test=bool(TEST_PATH_RE.search(mod.path)), mod=mod.name, parent=None, ast=mod.tree,
                       legacy_id=mod.name)
        mod_id = self.order[-1]
        mod.node_id = mod_id
        if mod.tree is None:
            return
        for d in _iter_defs(mod.tree.body):
            self._add_def(d, mod, mod_id, "", None, "")

    def _add_def(self, d: ast.stmt, mod: _Mod, parent_id: str, prefix: str, cls_id: str | None, lprefix: str) -> None:
        assert isinstance(d, _DEF) and mod.path is not None
        qual = f"{prefix}.{d.name}" if prefix else d.name
        lqual = f"{lprefix}.{d.name}" if lprefix else d.name
        kind = "class" if isinstance(d, ast.ClassDef) else "async_function" if isinstance(d, ast.AsyncFunctionDef) else "function"
        start = min([d.lineno] + [x.lineno for x in d.decorator_list])
        end = d.end_lineno or d.lineno
        lines = self.files[mod.path].splitlines()
        if kind == "class":
            first_child = min((c.lineno for c in d.body if isinstance(c, _DEF)), default=end + 1)
            text = "\n".join(lines[start - 1:min(end, first_child - 1)])
        else:
            text = "\n".join(lines[start - 1:end])
        node = self._add_node(id=self._unique_id(f"{mod.name}.{qual}"), qualname=qual, kind=kind, path=mod.path,
                              lineno=start, end_lineno=end, text=text[:MAX_TEXT],
                              is_test=bool(TEST_PATH_RE.search(mod.path)), mod=mod.name,
                              parent=parent_id, ast=d, cls=cls_id, legacy_id=f"{mod.name}.{lqual}")
        self._edge("contains", parent_id, node.id)
        parent = self.nodes[parent_id]
        if parent.kind == "module":
            mod.defs[d.name] = node.id
        else:
            parent.members[d.name] = node.id
        for sub in _iter_defs(d.body):
            self._add_def(sub, mod, node.id, qual, node.id if kind == "class" else None,
                          lqual if kind == "class" else lprefix)

    def _edge(self, etype: str, src: str, dst: str) -> None:
        if src != dst:
            self.edges[(etype, src, dst)] = None

    # ---- imports and module lookup ---------------------------------------------------------

    def _abs_module(self, mod: _Mod, level: int, module: str | None) -> str:
        if level == 0:
            return module or ""
        base = mod.name.split(".") if mod.is_pkg else mod.name.split(".")[:-1]
        if level > 1:
            base = base[:len(base) - (level - 1)]
        return ".".join(base + ([module] if module else []))

    def _collect_imports(self, mod: _Mod) -> None:
        assert mod.tree is not None
        mod.imports = self._imports_of(mod, mod.tree.body)
        if mod.node_id is None:
            return
        for n in ast.walk(mod.tree):  # every import in the file, including the ones inside functions
            targets: list[str] = []
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = self._abs_module(mod, n.level, n.module)
                targets = [base] + [f"{base}.{a.name}" if base else a.name for a in n.names]
            for t in targets:
                m = self._find_module(t)
                if m is not None and m.path is not None and m.name != mod.name:
                    self._edge("imports", mod.node_id, m.node_id)

    def _imports_of(self, mod: _Mod, body: Iterable[ast.AST]) -> dict[str, tuple[str, str | None]]:
        out: dict[str, tuple[str, str | None]] = {}
        for n in _walk_scope(body):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        out.setdefault(a.asname, (a.name, None))
                    else:
                        top = a.name.split(".")[0]
                        out.setdefault(top, (top, None))
            elif isinstance(n, ast.ImportFrom):
                base = self._abs_module(mod, n.level, n.module)
                for a in n.names:
                    if a.name != "*":
                        out.setdefault(a.asname or a.name, (base, a.name))
        return out

    def _find_module(self, name: str) -> _Mod | None:
        parts = name.split(".")
        if len(parts) > 1 and parts[0] in ROOT_PREFIXES:
            name = ".".join(parts[1:])
        m = self.mods.get(name)
        if m is not None:
            return m
        if "." not in name or name.split(".")[0] in sys.stdlib_module_names:
            return None  # suffix matching is only for dotted, non-stdlib names
        if self._suffix is None:
            self._suffix = defaultdict(list)
            for n in self.mods:
                parts = n.split(".")
                for i in range(1, len(parts)):
                    self._suffix[".".join(parts[i:])].append(n)
        cands = self._suffix.get(name, [])
        return self.mods[cands[0]] if len(cands) == 1 else None

    def _resolve_import(self, imp: tuple[str, str | None], depth: int) -> Target | None:
        modname, symbol = imp
        if depth > 6:
            return None
        m = self._find_module(modname)
        if symbol is None:
            return ("mod", m.name) if m else ("ext",)
        if m is None:
            return ("ext",)
        r = self._resolve_in_module(m, symbol, depth + 1)
        if r:
            return r
        sub = self._find_module(f"{m.name}.{symbol}")
        return ("mod", sub.name) if sub else None

    def _resolve_in_module(self, mod: _Mod, name: str, depth: int = 0) -> Target | None:
        if name in mod.defs:
            return ("node", mod.defs[name])
        if name in mod.imports:
            return self._resolve_import(mod.imports[name], depth)
        if mod.vt_state == 0:
            self._compute_module_vars(mod)
        if name in mod.var_types:
            return ("inst", mod.var_types[name])
        return None

    # ---- classes -----------------------------------------------------------------------------

    def _resolve_bases(self, nid: str) -> None:
        n = self.nodes[nid]
        if n.kind != "class":
            return
        ctx = self._ctx_of(self._scope_parent(nid))
        out: list[str] = []
        assert isinstance(n.ast, ast.ClassDef)
        for b in n.ast.bases:
            t = self._target(b, ctx)
            if t and t[0] == "node" and self.nodes[t[1]].kind == "class" and t[1] != nid:
                out.append(t[1])
                self._edge("inherits", nid, t[1])
        self.bases[nid] = out

    def _mro(self, cls_id: str) -> list[str]:
        out: list[str] = []
        stack = [cls_id]
        while stack:
            c = stack.pop(0)
            if c in out:
                continue
            out.append(c)
            stack = self.bases.get(c, []) + stack
        return out

    def _class_member(self, cls_id: str, name: str, skip_self: bool = False) -> str | None:
        for c in self._mro(cls_id)[1 if skip_self else 0:]:
            if name in self.nodes[c].members:
                return self.nodes[c].members[name]
        return None

    def _attr_type(self, cls_id: str, attr: str) -> str | None:
        for c in self._mro(cls_id):
            t = self._attr_types_of(c).get(attr)
            if t:
                return t
        return None

    def _attr_types_of(self, cls_id: str) -> dict[str, str]:
        if cls_id in self.attr_types:
            return self.attr_types[cls_id]
        if cls_id in self.attr_running:
            return {}
        self.attr_running.add(cls_id)
        out: dict[str, str] = {}
        cls = self.nodes[cls_id]
        assert isinstance(cls.ast, ast.ClassDef)
        cctx = self._ctx_of(self._scope_parent(cls_id))
        for st in cls.ast.body:  # class-level annotations and instances
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                t = self._ann_class(st.annotation, cctx)
                if t:
                    out[st.target.id] = t
        for mid in cls.members.values():
            m = self.nodes[mid]
            if m.kind == "class" or not isinstance(m.ast, _FUNC):
                continue
            mctx = self._ctx_of(mid)
            for n in _walk_scope(m.ast.body):
                tgt, val, ann = None, None, None
                if isinstance(n, ast.Assign) and len(n.targets) == 1:
                    tgt, val = n.targets[0], n.value
                elif isinstance(n, ast.AnnAssign):
                    tgt, val, ann = n.target, n.value, n.annotation
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    t = self._ann_class(ann, mctx) if ann is not None else None
                    if t is None and val is not None:
                        r = self._target(val, mctx)
                        t = r[1] if r and r[0] == "inst" else None
                    if t:
                        out.setdefault(tgt.attr, t)
        self.attr_running.discard(cls_id)
        if self.bases_ready:  # results computed against a partial MRO are not cached
            self.attr_types[cls_id] = out
        return out

    def _ann_class(self, ann: ast.AST | None, ctx: _Ctx) -> str | None:
        """Class id named by an annotation (handles Optional/Union/`X | None` and string annotations)."""
        if ann is None:
            return None
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                return self._ann_class(ast.parse(ann.value, mode="eval").body, ctx)
            except SyntaxError:
                return None
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            return self._ann_class(ann.left, ctx) or self._ann_class(ann.right, ctx)
        if isinstance(ann, ast.Subscript):
            base = _dotted(ann.value) or ""
            if base.split(".")[-1] in ("Optional", "Union", "Annotated", "Type", "type"):
                inner = ann.slice
                elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
                for e in elts:
                    r = self._ann_class(e, ctx)
                    if r:
                        return r
            return None
        if isinstance(ann, (ast.Name, ast.Attribute)):
            t = self._target(ann, ctx)
            if t and t[0] == "node" and self.nodes[t[1]].kind == "class":
                return t[1]
        return None

    # ---- scopes --------------------------------------------------------------------------------

    def _scope_parent(self, nid: str) -> str:
        """Nearest enclosing function or module (class bodies are not visible to nested functions)."""
        p = self.nodes[nid].parent
        while p is not None and self.nodes[p].kind == "class":
            p = self.nodes[p].parent
        assert p is not None, nid
        return p

    def _ctx_of(self, nid: str) -> _Ctx:
        if nid in self.ctxs:
            return self.ctxs[nid]
        n = self.nodes[nid]
        mod = self.mods[n.mod]
        if n.kind == "module":
            ctx = _Ctx(mod, nid, None, {}, mod.imports, {}, set())
            self.ctxs[nid] = ctx
            return ctx
        base = self._ctx_of(self._scope_parent(nid))
        ctx = _Ctx(mod, nid, None if n.kind == "class" else (n.cls or base.owner_cls),
                   dict(base.local_defs), dict(base.imports), dict(base.var_types), set())
        self.ctxs[nid] = ctx  # registered early: recursive type inference sees a partial context
        assert isinstance(n.ast, _DEF)
        if isinstance(n.ast, _FUNC):
            a = n.ast.args
            params = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
            ctx.bound.update(p.arg for p in params)
            for p in params:
                t = self._ann_class(p.annotation, ctx)
                if t:
                    ctx.var_types[p.arg] = t
            ctx.local_defs.update(n.members)
            ctx.imports.update(self._imports_of(mod, n.ast.body))
        self._scan_assignments(n.ast.body, ctx)
        return ctx

    def _scan_assignments(self, body: Iterable[ast.AST], ctx: _Ctx, store: dict[str, str] | None = None) -> None:
        store = ctx.var_types if store is None else store
        for n in _walk_scope(body):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                ctx.bound.add(n.id)
            if isinstance(n, ast.Assign):
                names = [t.id for t in n.targets if isinstance(t, ast.Name)]
                if names:
                    r = self._target(n.value, ctx)
                    for nm in names:
                        if r and r[0] == "inst":
                            store[nm] = r[1]
                        else:
                            store.pop(nm, None)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                t = self._ann_class(n.annotation, ctx)
                if t is None and n.value is not None:
                    r = self._target(n.value, ctx)
                    t = r[1] if r and r[0] == "inst" else None
                if t:
                    store[n.target.id] = t
            elif isinstance(n, ast.With) or isinstance(n, ast.AsyncWith):
                for item in n.items:
                    if isinstance(item.optional_vars, ast.Name):
                        r = self._target(item.context_expr, ctx)
                        if r and r[0] == "inst":
                            store[item.optional_vars.id] = r[1]

    def _compute_module_vars(self, mod: _Mod) -> None:
        mod.vt_state = 1
        if mod.tree is not None and mod.node_id is not None:
            self._scan_assignments(mod.tree.body, self._ctx_of(mod.node_id), store=mod.var_types)
        mod.vt_state = 2

    # ---- expression targets ---------------------------------------------------------------------

    def _lookup_name(self, name: str, ctx: _Ctx) -> Target | None:
        if name in ctx.var_types:
            return ("inst", ctx.var_types[name])
        if name in ("self", "cls") and ctx.owner_cls:
            return ("inst", ctx.owner_cls)
        if name in ctx.local_defs:
            return ("node", ctx.local_defs[name])
        in_module_scope = self.nodes[ctx.node_id].kind == "module"
        if name in ctx.bound and not in_module_scope:
            return None  # a parameter or local variable of unknown type shadows module names
        if name in ctx.imports and not in_module_scope:
            return self._resolve_import(ctx.imports[name], 0)
        return self._resolve_in_module(ctx.mod, name)

    def _target(self, expr: ast.AST, ctx: _Ctx) -> Target | None:
        if isinstance(expr, ast.Name):
            return self._lookup_name(expr.id, ctx)
        if isinstance(expr, ast.Await):
            return self._target(expr.value, ctx)
        if isinstance(expr, ast.Attribute):
            base = self._target(expr.value, ctx)
            if base is None:
                return None
            kind, ref = base[0], base[1] if len(base) > 1 else None
            if kind == "ext":
                return ("ext",)
            if kind == "mod":
                m = self.mods[ref]
                r = self._resolve_in_module(m, expr.attr)
                if r:
                    return r
                sub = self._find_module(f"{m.name}.{expr.attr}")
                return ("mod", sub.name) if sub else None
            if kind == "node":
                if self.nodes[ref].kind == "class":
                    mid = self._class_member(ref, expr.attr)
                    return ("node", mid) if mid else None
                return None
            if kind == "inst":
                mid = self._class_member(ref, expr.attr)
                if mid:
                    return ("node", mid)
                t = self._attr_type(ref, expr.attr)
                return ("inst", t) if t else None
            if kind == "super":
                mid = self._class_member(ref, expr.attr, skip_self=True)
                return ("node", mid) if mid else None
            return None
        if isinstance(expr, ast.Call):
            f = expr.func
            if isinstance(f, ast.Name) and f.id == "super" and ctx.owner_cls and "super" not in ctx.imports:
                return ("super", ctx.owner_cls)
            t = self._target(f, ctx)
            if t is None:
                return None
            if t[0] == "ext":
                return ("ext",)
            if t[0] == "node":
                n = self.nodes[t[1]]
                if n.kind == "class":
                    return ("inst", n.id)
                if isinstance(n.ast, _FUNC) and n.ast.returns is not None:
                    c = self._ann_class(n.ast.returns, self._ctx_of(self._scope_parent(n.id)))
                    return ("inst", c) if c else None
            return None
        return None

    # ---- pass 2: edges ---------------------------------------------------------------------------

    def _methods_named(self, name: str) -> list[str]:
        if self._method_index is None:
            idx: dict[str, list[str]] = defaultdict(list)
            for nid, n in self.nodes.items():
                if n.kind != "module" and n.cls is not None and n.kind != "class":
                    idx[n.qualname.rsplit(".", 1)[-1]].append(nid)
            self._method_index = idx
        return self._method_index.get(name, [])

    def _emit_scope_edges(self, nid: str) -> None:
        n = self.nodes[nid]
        if n.ast is None:
            return
        ctx = self._ctx_of(nid)
        for node in _walk_scope(n.ast.body):
            if isinstance(node, _DEF):
                # Decorators run in this scope, applied to the decorated definition.
                for dec in node.decorator_list:
                    if not isinstance(dec, ast.Call):
                        self._link_call(nid, self._target(dec, ctx))
            elif isinstance(node, ast.Call):
                t = self._target(node.func, ctx)
                self._link_call(nid, t)
                if t is None and isinstance(node.func, ast.Attribute):
                    self._maybe_call(nid, node.func.attr, n.is_test)
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, (ast.Name, ast.Attribute)):
                        r = self._target(arg, ctx)
                        if r and r[0] == "node":
                            self._edge("references", nid, r[1])

    def _link_call(self, src: str, t: Target | None) -> None:
        if not t or t[0] != "node":
            return
        tgt = self.nodes[t[1]]
        self._edge("calls", src, tgt.id)
        if tgt.kind == "class":
            init = self._class_member(tgt.id, "__init__")
            if init:
                self._edge("calls", src, init)

    def _maybe_call(self, src: str, name: str, caller_is_test: bool) -> None:
        if name.startswith("__") or len(name) < 4:
            return
        cands = [c for c in self._methods_named(name) if caller_is_test or not self.nodes[c].is_test]
        if len(cands) == 1:
            self._edge("maybe_calls", src, cands[0])

    # ---- export ----------------------------------------------------------------------------------

    def to_json(self, repo_name: str = "") -> dict:
        nodes = [{"id": n.id, "name": n.id, "text": n.text, "path": n.path, "qualname": n.qualname,
                  "kind": n.kind, "legacy_id": n.legacy_id, "lineno": n.lineno, "end_lineno": n.end_lineno, "is_test": n.is_test}
                 for n in (self.nodes[i] for i in self.order)]
        edges = [{"type": t, "source": s, "target": d, "key": 0} for (t, s, d) in self.edges]
        return {"directed": True, "multigraph": True, "graph": {"repo_name": repo_name, "builder": "g4a.graph_builder"},
                "nodes": nodes, "edges": edges}


def build_graph(files: dict[str, str], repo_name: str = "") -> dict:
    """Build the repaired graph from {path: source}. Files that fail to parse still get a module node."""
    b = _Builder(files)
    b.run()
    return b.to_json(repo_name)


def build_graph_from_snapshot(snapshot: Path | IO[bytes], repo_name: str = "") -> dict:
    return build_graph(read_python_files(snapshot), repo_name)
