"""Key-quality diagnostics for the flat and dichotomous keys, on the external set and on train.

Usage: uv run python -m experiments.s3_external_diag   -> results/s3_external/diag.json
(a) first-couplet errors of the dichotomous key and flat rescues, (b) candidate ceilings, (c) unparsed picks.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from experiments import s2_key, s3_external as s3
from experiments.s2_localize import DATA, ZIP, load_index as train_index
from g4a import external as ext
from g4a.key_localize import _parse_list
from g4a.localize import files_from_units

N_FLAT, N_FILES, K_FILES = 20, 10, 2


def diagnose(rows: list[dict], retrieval: dict[str, list[str]], gold: dict[str, tuple[set, set]]) -> dict:
    """rows: key result rows of one variant; gold[tid] = (gold files, gold units)."""
    by = {(r["instance_id"], r["mode"]): r for r in rows}
    tids = sorted(t for t in retrieval if (t, "flat") in by and (t, "dichotomous") in by)
    out = dict(n=len(tids), first_couplet_wrong_goldfile=0, wrong_goldfile_flat_unit1=0, first_couplet_wrong=0, wrong_flat_unit1=0, wrong_flat_ids=[], dich_unit1_in_wrong=0,
               flat_ceiling_missing=0, dich_file_missing=0, dich_file_missing_ids=[],
               flat_picks0=0, flat_picks0_unit1=0, dich_picks0=0, dich_picks0_unit1=0,
               dich_first_unparsed=0, dich_first_unparsed_unit1=0)
    for t in tids:
        gf, gu = gold[t]
        rk = retrieval[t]
        flat, dich = by[(t, "flat")], by[(t, "dichotomous")]
        if not (set(rk[:N_FLAT]) & gu):
            out["flat_ceiling_missing"] += 1
        files: list[str] = []
        for uid in rk:
            p = uid.split("::", 1)[0]
            if p not in files:
                files.append(p)
            if len(files) == N_FILES:
                break
        if not (set(files) & gf):
            out["dich_file_missing"] += 1
            out["dich_file_missing_ids"].append(t)
        parsed = _parse_list(dich["raw"][0], files)[:K_FILES]
        chosen = parsed or files[:1]
        f1 = bool(set(flat["ranking"][:1]) & gu)
        d1 = bool(set(dich["ranking"][:1]) & gu)
        if not parsed:
            out["dich_first_unparsed"] += 1
            out["dich_first_unparsed_unit1"] += d1
        guf = {u.split("::", 1)[0] for u in gu}  # files of the gold units (the definition used in the S2 notes)
        if not (set(chosen) & gf):
            out["first_couplet_wrong_goldfile"] += 1
            out["wrong_goldfile_flat_unit1"] += f1
        if not (set(chosen) & guf):
            out["first_couplet_wrong"] += 1
            out["wrong_flat_unit1"] += f1
            out["dich_unit1_in_wrong"] += d1
            out["wrong_flat_ids"].append(t)
        if not flat["picks"]:
            out["flat_picks0"] += 1
            out["flat_picks0_unit1"] += f1
        if not dich["picks"]:
            out["dich_picks0"] += 1
            out["dich_picks0_unit1"] += d1
    out["flat_ceiling"] = round(1 - out["flat_ceiling_missing"] / len(tids), 3)
    out["flat_unit1"] = round(sum(bool(set(by[(t, "flat")]["ranking"][:1]) & gold[t][1]) for t in tids) / len(tids), 3)
    return out


def external() -> dict:
    tasks = {t["instance_id"]: t for t in ext.load_tasks()}
    truths = {r["instance_id"]: r for r in s3.load_jsonl(ext.TRUTH_PATH)}
    keys = s3.load_jsonl(s3.OUT / "keys.jsonl")
    res = {}
    for v in s3.VARIANTS:
        retr = {r["instance_id"]: r["ranking"] for r in s3.load_jsonl(s3.OUT / f"retrieval_{v}.jsonl")}
        rows = [r for r in keys if r["variant"] == v]
        ids = {r["instance_id"] for r in rows}
        gold = {t: s3.gold_for(truths[t], s3.load_index(tasks[t])) for t in ids}
        res[v] = diagnose(rows, {t: retr[t] for t in ids}, gold)
    return res


def train() -> dict:
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines()}
    truths = {r["instance_id"]: r for r in s2_key.load_jsonl(Path("data/derived/groundtruth.jsonl"))}
    res = {}
    with zipfile.ZipFile(ZIP) as zf:
        for v, d in (("hybrid", Path("results/s2_key")), ("bm25", Path("results/s2_key_bm25"))):
            retr = {r["instance_id"]: r["ranking"] for r in s2_key.load_jsonl(d / "retrieval.jsonl")}
            rows = s2_key.load_jsonl(d / "keys.jsonl")
            ids = {r["instance_id"] for r in rows}
            gold = {t: s2_key.gold_for(tasks[t], truths[t], train_index(tasks[t], zf)) for t in ids}
            res[v] = diagnose(rows, {t: retr[t] for t in ids}, gold)
    return res


def main() -> None:
    out = {"external": external(), "train": train()}
    p = s3.OUT / "diag.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    for ds, vs in out.items():
        for v, d in vs.items():
            print(ds, v, {k: x for k, x in d.items() if not k.endswith("_ids")})


if __name__ == "__main__":
    main()
