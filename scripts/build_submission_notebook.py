"""Turn submission/<name>/ into a Kaggle notebook that writes /kaggle/working/submission.zip.

The bundle files are embedded as base64 so the notebook needs no datasets and no internet.
Usage:
    uv run python scripts/build_submission_notebook.py submission/baseline_v0 --kaggle-user <user>
Then:
    kaggle kernels push -p build/kernels/<slug>
    kaggle competitions submit gemma-4-developer-agent -k <user>/<slug> -f submission.zip -v <version> -m "<message>"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

COMPETITION = "gemma-4-developer-agent"
ALLOWED_SUFFIXES = {".yaml", ".yml", ".md", ".py", ".sh", ".json", ".safetensors", ".txt"}


def collect(bundle: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            raise SystemExit(f"Disallowed file type in bundle: {path}")
        files[path.relative_to(bundle).as_posix()] = base64.b64encode(path.read_bytes()).decode()
    if "agent.yaml" not in files:
        raise SystemExit("agent.yaml must sit at the bundle root")
    return files


def notebook(files: dict[str, str], name: str, digest: str) -> dict:
    payload = json.dumps(files)
    code = f'''import base64, json, zipfile
from pathlib import Path

# Bundle "{name}" (sha256 {digest}), generated from the project repository.
FILES = json.loads({payload!r})

out = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel, b64 in FILES.items():
        zf.writestr(rel, base64.b64decode(b64))
print(out, out.stat().st_size, "bytes,", len(FILES), "files")
for rel in FILES:
    print(" ", rel)
'''
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# Submission bundle `{name}`\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("submission/build/kernels"))
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    name = bundle.name
    files = collect(bundle)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    slug = f"g4a-{name.replace('_', '-')}"

    out = args.out_root / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook(files, name, digest), indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.kaggle_user}/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    print(f"{out}  ({len(files)} files, sha256 {digest})")


if __name__ == "__main__":
    main()
