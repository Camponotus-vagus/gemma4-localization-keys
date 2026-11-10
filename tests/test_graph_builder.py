"""Sanity checks for g4a.graph_builder on a tiny synthetic repository.

Run: uv run python tests/test_graph_builder.py   (pytest-compatible, but pytest is not required)
"""

from __future__ import annotations

from g4a.graph_builder import build_graph

FILES = {
    "pkg/__init__.py": "from .core import Engine\n",
    "pkg/util.py": (
        "def helper(x):\n    return x\n\n"
        "def deco(fn):\n    return fn\n"
    ),
    "pkg/base.py": (
        "class Base:\n"
        "    def run(self):\n        return 1\n"
    ),
    "pkg/core.py": (
        "import pkg.util\n"
        "from pkg.util import helper, deco\n"
        "from .base import Base\n\n"
        "class Engine(Base):\n"
        "    def __init__(self):\n        self.part = Part()\n\n"
        "    @deco\n"
        "    async def start(self):\n"
        "        self.prepare()\n"
        "        self.part.spin()\n"
        "        return helper(1)\n\n"
        "    def prepare(self):\n        pkg.util.helper(2)\n        return super().run()\n\n"
        "class Part:\n"
        "    def spin(self):\n        return 0\n\n"
        "def build():\n    e = Engine()\n    e.prepare()\n    return e\n\n"
        "VALUE = build()\n"
    ),
    "tests/test_core.py": "from pkg import Engine\n\ndef test_it():\n    Engine().prepare()\n",
    "broken.py": "def oops(:\n",
}


def _graph() -> tuple[dict, set[tuple[str, str, str]], dict[str, dict]]:
    g = build_graph(FILES)
    edges = {(e["type"], e["source"], e["target"]) for e in g["edges"]}
    return g, edges, {n["id"]: n for n in g["nodes"]}


def test_nodes_and_kinds() -> None:
    _, _, nodes = _graph()
    assert nodes["pkg.core.Engine.start"]["kind"] == "async_function"
    assert nodes["pkg.core.Engine"]["kind"] == "class"
    assert nodes["pkg.core"]["kind"] == "module" and nodes["pkg.core"]["path"] == "pkg/core.py"
    assert nodes["pkg.core.Engine.start"]["lineno"] == 9  # decorator line is included
    assert nodes["tests.test_core.test_it"]["is_test"] and not nodes["pkg.core.build"]["is_test"]
    assert "broken" in nodes  # unparsable files still get a module node


def test_contains_and_imports() -> None:
    _, edges, _ = _graph()
    assert ("contains", "pkg.core", "pkg.core.Engine") in edges
    assert ("contains", "pkg.core.Engine", "pkg.core.Engine.start") in edges
    assert ("imports", "pkg.core", "pkg.util") in edges
    assert ("imports", "pkg.core", "pkg.base") in edges  # relative import
    assert ("imports", "pkg", "pkg.core") in edges


def test_calls_resolution() -> None:
    _, edges, _ = _graph()
    assert ("calls", "pkg.core.Engine.start", "pkg.core.Engine.prepare") in edges  # self.method
    assert ("calls", "pkg.core.Engine.start", "pkg.core.Part.spin") in edges  # typed attribute
    assert ("calls", "pkg.core.Engine.start", "pkg.util.helper") in edges  # from-import
    assert ("calls", "pkg.core.Engine.prepare", "pkg.util.helper") in edges  # module attribute
    assert ("calls", "pkg.core.Engine.prepare", "pkg.base.Base.run") in edges  # super() through the MRO
    assert ("calls", "pkg.core.build", "pkg.core.Engine") in edges  # constructor
    assert ("calls", "pkg.core.build", "pkg.core.Engine.__init__") in edges
    assert ("calls", "pkg.core.build", "pkg.core.Engine.prepare") in edges  # local variable type
    assert ("calls", "pkg.core", "pkg.core.build") in edges  # module-level call
    assert ("calls", "pkg.core.Engine", "pkg.util.deco") in edges  # decorator, evaluated in the class scope
    assert ("calls", "tests.test_core.test_it", "pkg.core.Engine.prepare") in edges  # via re-export in __init__


def test_inherits() -> None:
    _, edges, _ = _graph()
    assert ("inherits", "pkg.core.Engine", "pkg.base.Base") in edges


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
