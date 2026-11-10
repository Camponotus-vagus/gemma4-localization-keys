"""LLM localization framed as identification keys.

Two ways to let a model pick the edit location among retrieved candidates:

- multi-access key ("flat"): the model sees the top units from several files at once and picks.
- dichotomous key ("hierarchical"): the model first picks files from compact file descriptions
  (their top-level symbols), then picks units inside the chosen files.

Both return a full unit ranking: the model's picks first, then the retrieval order, so that
hit@k is comparable with retrieval-only baselines.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field

from g4a.repo_index import RepoIndex, Unit

ISSUE_CHARS = 5000


@dataclass
class KeyResult:
    ranking: list[str]
    picks: list[str]
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    raw: list[str] = field(default_factory=list)


def _first_doc_line(text: str) -> str:
    m = re.search(r'"""\s*(.+?)\s*(?:\n|""")', text) or re.search(r"'''\s*(.+?)\s*(?:\n|''')", text)
    return m.group(1)[:100] if m else ""


def _signature(u: Unit) -> str:
    for line in u.text.splitlines():
        s = line.strip()
        if s.startswith(("def ", "async def ", "class ")):
            return s[:160]
    return u.qualname


def unit_card(u: Unit) -> str:
    doc = _first_doc_line(u.text)
    calls = ", ".join(u.calls[:8])
    parts = [f"{u.path}::{u.qualname}", f"  {_signature(u)}"]
    if doc:
        parts.append(f"  doc: {doc}")
    if calls:
        parts.append(f"  calls: {calls}")
    return "\n".join(parts)


def file_card(idx: RepoIndex, path: str, limit: int = 25, rank: dict[str, int] | None = None) -> str:
    """Compact description of a file: its top-level symbols, most retrieval-relevant first."""
    tops = [u for u in idx.units if u.path == path and "." not in u.qualname]
    if rank:
        # A top-level symbol inherits the best rank of itself or any of its methods.
        best = {u.qualname: rank.get(u.uid, 10**9) for u in tops}
        for uid, r in rank.items():
            p, q = uid.split("::", 1)
            if p == path and "." in q:
                head = q.split(".", 1)[0]
                if head in best:
                    best[head] = min(best[head], r)
        tops = sorted(tops, key=lambda u: best[u.qualname])
    methods = [u for u in idx.units if u.path == path and "." in u.qualname]
    names = [("class " if u.kind == "class" else "def ") + u.qualname for u in tops[:limit]]
    extra = len(tops) - len(names)
    try:
        mod_doc = ast.get_docstring(ast.parse(idx.files[path])) or ""
    except SyntaxError:
        mod_doc = ""
    head = f"{path} ({len(tops)} top-level symbols, {len(methods)} methods)"
    body = "; ".join(names) + (f"; ... +{extra} more" if extra > 0 else "")
    doc = f"\n  doc: {mod_doc.splitlines()[0][:120]}" if mod_doc else ""
    return f"{head}{doc}\n  {body}"


SYSTEM = (
    "You locate where a code change must be made to resolve a GitHub issue. "
    "Answer with JSON only, no prose."
)


def _parse_list(text: str, allowed: list[str]) -> list[str]:
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    picks: list[str] = []
    if m:
        try:
            obj = json.loads(m.group(0))
            items = obj.get("choices", obj.get("answer", [])) if isinstance(obj, dict) else obj
            picks = [str(x) for x in items] if isinstance(items, list) else [str(items)]
        except json.JSONDecodeError:
            pass
    if not picks:  # fall back to any candidate id or number mentioned
        picks = re.findall(r"\b\d+\b", text)
    out: list[str] = []
    for p in picks:
        p = p.strip()
        if p.isdigit() and 1 <= int(p) <= len(allowed):
            p = allowed[int(p) - 1]
        if p in allowed and p not in out:
            out.append(p)
    return out


class KeyLocalizer:
    def __init__(self, model_id: str = "mlx-community/gemma-4-31b-it-4bit", max_tokens: int = 200) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self.max_tokens = max_tokens

    def _ask(self, user: str, res: KeyResult) -> str:
        from mlx_lm import generate

        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        t0 = time.perf_counter()
        out = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens, verbose=False)
        res.seconds += time.perf_counter() - t0
        res.calls += 1
        res.prompt_tokens += len(self.tokenizer.encode(prompt))
        res.completion_tokens += len(self.tokenizer.encode(out))
        res.raw.append(out)
        return out

    @staticmethod
    def _complete(picks: list[str], retrieval: list[str]) -> list[str]:
        return picks + [u for u in retrieval if u not in picks]

    def flat(self, idx: RepoIndex, issue: str, retrieval: list[str], n: int = 20, k: int = 3) -> KeyResult:
        by_uid = {u.uid: u for u in idx.units}
        cands = retrieval[:n]
        listing = "\n".join(f"[{i}] {unit_card(by_uid[c])}" for i, c in enumerate(cands, 1))
        user = (
            f"Issue:\n{issue[:ISSUE_CHARS]}\n\nCandidate code units:\n{listing}\n\n"
            f"Which units must be edited to resolve the issue? Pick at most {k}, most likely first. "
            'Reply as {"choices": [numbers]}.'
        )
        res = KeyResult(ranking=[], picks=[])
        res.picks = _parse_list(self._ask(user, res), cands)[:k]
        res.ranking = self._complete(res.picks, retrieval)
        return res

    def dichotomous(self, idx: RepoIndex, issue: str, retrieval: list[str],
                    n_files: int = 10, k_files: int = 2, n_units: int = 15, k: int = 3) -> KeyResult:
        by_uid = {u.uid: u for u in idx.units}
        files: list[str] = []
        for uid in retrieval:
            p = uid.split("::", 1)[0]
            if p not in files:
                files.append(p)
            if len(files) == n_files:
                break
        rank = {uid: i for i, uid in enumerate(retrieval)}
        listing = "\n".join(f"[{i}] {file_card(idx, p, rank=rank)}" for i, p in enumerate(files, 1))
        user = (
            f"Issue:\n{issue[:ISSUE_CHARS]}\n\nCandidate files:\n{listing}\n\n"
            f"Which files contain the code that must change? Pick at most {k_files}, most likely first. "
            'Reply as {"choices": [numbers]}.'
        )
        res = KeyResult(ranking=[], picks=[])
        chosen_files = _parse_list(self._ask(user, res), files)[:k_files] or files[:1]

        # Second couplet: units inside the chosen files, in retrieval order.
        cands = [u for u in retrieval if u.split("::", 1)[0] in chosen_files][:n_units]
        listing = "\n".join(f"[{i}] {unit_card(by_uid[c])}" for i, c in enumerate(cands, 1))
        user = (
            f"Issue:\n{issue[:ISSUE_CHARS]}\n\nCode units in {', '.join(chosen_files)}:\n{listing}\n\n"
            f"Which units must be edited? Pick at most {k}, most likely first. "
            'Reply as {"choices": [numbers]}.'
        )
        res.picks = _parse_list(self._ask(user, res), cands)[:k]
        # Keep the file decision even when no unit is picked: chosen files' units come first.
        in_files = [u for u in retrieval if u.split("::", 1)[0] in chosen_files]
        res.ranking = self._complete(res.picks + [u for u in in_files if u not in res.picks], retrieval)
        return res
