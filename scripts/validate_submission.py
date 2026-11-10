"""Compile a submission directory the way the scorer does, without a model server or sandbox.

Catches YAML, !include, tool-name, limit and generation-constraint errors before
spending a daily submission.
Usage: uv run python scripts/validate_submission.py submission/baseline_v0
"""

from __future__ import annotations

import sys
from pathlib import Path

from adk_submission import ModelRegistry
from adk_submission.compiler import compile_submission
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import build_submission_limits

MODEL = "gemma-4-31b-it-qat-w4a16-ct"


# Stubs with the same names and signatures as the harness tools.
def run_command(command: str) -> str: return ""
def submit_patch() -> str: return ""
def get_status() -> str: return ""
def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str: return ""
def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str: return ""
def write_file(filepath: str, content: str) -> str: return ""
def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str: return ""
def search_similar_code(query: str, k: int = 10) -> str: return ""
def get_code_subgraph(nodes: list[str]) -> str: return ""


TOOLS = {f.__name__: f for f in (
    run_command, submit_patch, get_status, read_file, edit_file, write_file,
    get_code_neighbors, search_similar_code, get_code_subgraph,
)}


def describe(agent, depth: int = 0) -> None:
    tools = [getattr(t, "name", None) or getattr(t, "__name__", type(t).__name__) for t in getattr(agent, "tools", [])]
    model = getattr(getattr(agent, "model", None), "model", getattr(agent, "model", None))
    print(f"{'  ' * depth}- {agent.name} [{type(agent).__name__}] model={model} tools={tools}")
    for sub in getattr(agent, "sub_agents", []) or []:
        describe(sub, depth + 1)


def main() -> None:
    bundle = Path(sys.argv[1])
    models = ModelRegistry()
    models.register(MODEL, LiteLlm(model=f"openai/{MODEL}", api_base="http://127.0.0.1:1/v1", api_key="EMPTY"))
    limits, constraints = build_submission_limits()
    agent = compile_submission(
        submission_dir=bundle,
        tool_registry=TOOLS,
        model_registry=models,
        limits=limits,
        generation_constraints=constraints,
    )
    print(f"OK: {bundle} compiles")
    describe(agent)


if __name__ == "__main__":
    main()
