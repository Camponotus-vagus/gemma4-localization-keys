"""Pin-aware wheel selection for local sandboxes.

swegemma keeps the newest wheel when several versions of a package are available, ignoring
the version constraints declared by the repository at base_commit. The public wheels ship
many versions of some packages (e.g. starlette 0.19 to 1.6), so older FastAPI commits get a
starlette that no longer accepts their arguments. This module keeps, for every constrained
dependency, only the newest wheel that satisfies the constraint.
"""

from __future__ import annotations

import hashlib
import tarfile
import tomllib
from collections import defaultdict
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def wheel_name_version(path: Path) -> tuple[str, Version] | None:
    parts = path.name.split("-")
    if len(parts) < 3:
        return None
    try:
        return canonicalize_name(parts[0]), Version(parts[1])
    except InvalidVersion:
        return None


def declared_requirements(pyproject_text: str) -> list[Requirement]:
    """Runtime and optional/dev dependencies declared in pyproject.toml (PEP 621 and PEP 735 groups)."""
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    raw: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        raw.extend(extra)
    for group in data.get("dependency-groups", {}).values():
        raw.extend(r for r in group if isinstance(r, str))
    reqs = []
    for r in raw:
        try:
            reqs.append(Requirement(r))
        except InvalidRequirement:
            continue
    return reqs


def select_wheels(wheel_paths: list[Path], requirements: list[Requirement]) -> list[Path]:
    """Drop wheels whose version violates a declared specifier, unless no version satisfies it."""
    by_pkg: dict[str, list[tuple[Version, Path]]] = defaultdict(list)
    unparsed: list[Path] = []
    for p in wheel_paths:
        nv = wheel_name_version(p)
        if nv is None:
            unparsed.append(p)
        else:
            by_pkg[nv[0]].append((nv[1], p))

    specs: dict[str, list] = defaultdict(list)
    for r in requirements:
        if r.specifier:
            specs[canonicalize_name(r.name)].append(r.specifier)

    kept: list[Path] = list(unparsed)
    for pkg, versions in by_pkg.items():
        allowed = [
            (v, p) for v, p in versions
            if all(s.contains(v, prereleases=True) for s in specs.get(pkg, []))
        ]
        # Keep every build of the chosen version (cp312 and cp313 wheels coexist).
        pool = allowed or versions
        best = max(v for v, _ in pool)
        kept.extend(p for v, p in pool if v == best)
    return sorted(kept)


def selection_digest(paths: list[Path]) -> str:
    return hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]


def materialize(selected: list[Path], root: Path) -> Path:
    """Hard-link a selection into root/<digest>/ and return that directory."""
    out = root / selection_digest(selected)
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for p in selected:
            dst = tmp / p.name
            if not dst.exists():
                dst.hardlink_to(p)
        tmp.rename(out)
    return out


def pyproject_from_snapshot(tgz: Path) -> str | None:
    with tarfile.open(tgz, "r:gz") as tf:
        for member in tf:
            if member.name in ("./pyproject.toml", "pyproject.toml") and member.isfile():
                fh = tf.extractfile(member)
                return fh.read().decode("utf-8", errors="replace") if fh else None
    return None

