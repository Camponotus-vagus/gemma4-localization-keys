"""Dense retrieval with a local text/code embedding model, cached by content hash.

Units that are identical across commits (most of them) are embedded once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from g4a.repo_index import RepoIndex, Unit

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = Path("data/derived/emb_cache")


def unit_text(u: Unit) -> str:
    return f"{u.path}\n{u.qualname}\n{u.text[:2000]}"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class EmbeddingRanker:
    def __init__(self, model_id: str = MODEL_ID, device: str = "mps", max_seq_length: int = 512) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.model.max_seq_length = max_seq_length
        slug = model_id.replace("/", "__")
        self.vec_path = CACHE_DIR / f"{slug}.npy"
        self.key_path = CACHE_DIR / f"{slug}.keys.json"
        self.keys: dict[str, int] = json.loads(self.key_path.read_text()) if self.key_path.exists() else {}
        self.vecs = np.load(self.vec_path) if self.vec_path.exists() else np.zeros((0, self.model.get_sentence_embedding_dimension()), np.float32)

    def ensure(self, texts: list[str], batch_size: int = 32) -> None:
        missing = list(dict.fromkeys(t for t in texts if _key(t) not in self.keys))
        if not missing:
            return
        new = self.model.encode(missing, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(missing) > 500).astype(np.float32)
        base = len(self.vecs)
        for i, t in enumerate(missing):
            self.keys[_key(t)] = base + i
        self.vecs = np.vstack([self.vecs, new])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.vecs)
        self.key_path.write_text(json.dumps(self.keys))

    def rank_units(self, idx: RepoIndex, issue: str) -> tuple[list[str], np.ndarray]:
        texts = [unit_text(u) for u in idx.units]
        self.ensure(texts)
        M = self.vecs[[self.keys[_key(t)] for t in texts]]
        q = self.model.encode([issue[:4000]], prompt_name="query", normalize_embeddings=True)[0]
        s = M @ q
        order = np.argsort(-s, kind="stable")
        return [idx.units[i].uid for i in order], s


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal rank fusion of several rankings over the same items."""
    score: dict[str, float] = {}
    for r in rankings:
        for pos, item in enumerate(r):
            score[item] = score.get(item, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda x: -score[x])
