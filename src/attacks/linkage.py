"""
Embedding linkage attacks (no attacker LLM needed).

b3-lite: identify the CUAD category (41-way) and the contract (N-way) from the
         transmitted questions by embedding similarity against a candidate pool.
b3-replay: the same, but the candidate texts are what the attacker obtains by
         replaying the public pipeline's topic step on each (contract, category).
c:       the same with the union of k query sets from one contract.

All rankings are "max over candidate chunks of cosine(mean query embedding, chunk)".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


@dataclass
class CandidateIndex:
    keys: list[Hashable]
    chunk_embs: np.ndarray           # (n_chunks, dim), L2-normalised
    chunk_owner: np.ndarray          # (n_chunks,) index into keys

    @property
    def n(self) -> int:
        return len(self.keys)


class EmbeddingLinker:
    def __init__(self, embed_model, chunk_chars: int = 1500, max_chunks: int = 6):
        self._embed = embed_model
        self._chunk_chars = chunk_chars
        self._max_chunks = max_chunks

    def _chunks(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return [""]
        out = [text[i:i + self._chunk_chars] for i in range(0, len(text), self._chunk_chars)]
        return out[: self._max_chunks]

    def build_index(self, candidates: dict[Hashable, str | list[str]]) -> CandidateIndex:
        keys = list(candidates)
        texts: list[str] = []
        owner: list[int] = []
        for i, k in enumerate(keys):
            v = candidates[k]
            chunks = list(v) if isinstance(v, list) else self._chunks(v)
            for c in chunks:
                texts.append(c)
                owner.append(i)
        embs = self._embed.encode(texts, normalize_embeddings=True, batch_size=64)
        return CandidateIndex(keys=keys, chunk_embs=np.asarray(embs), chunk_owner=np.asarray(owner))

    def query_embedding(self, questions: list[str]) -> np.ndarray:
        embs = self._embed.encode(questions, normalize_embeddings=True)
        q = np.asarray(embs).mean(axis=0)
        return q / (np.linalg.norm(q) + 1e-9)

    def rank(self, questions: list[str], index: CandidateIndex) -> list[tuple[Hashable, float]]:
        q = self.query_embedding(questions)
        sims = index.chunk_embs @ q
        best = np.full(index.n, -np.inf)
        np.maximum.at(best, index.chunk_owner, sims)
        order = np.argsort(-best)
        return [(index.keys[i], float(best[i])) for i in order]


def rank_of(true_key: Hashable, ranked: list[tuple[Hashable, float]]) -> int:
    for i, (k, _) in enumerate(ranked):
        if k == true_key:
            return i + 1
    return len(ranked) + 1


@dataclass
class LinkageRow:
    sample_id: str
    true_key: str
    rank: int
    n_candidates: int
    top1: bool = field(init=False)
    top5: bool = field(init=False)

    def __post_init__(self):
        self.top1 = self.rank == 1
        self.top5 = self.rank <= 5

    def to_dict(self) -> dict:
        return {"sample_id": self.sample_id, "true_key": self.true_key, "rank": self.rank,
                "n_candidates": self.n_candidates, "top1": self.top1, "top5": self.top5}
