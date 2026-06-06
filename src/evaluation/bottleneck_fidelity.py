"""
Test 3: Bottleneck Fidelity.

How well do the selected proxy questions span the semantic space of the original?

Metrics:
  - max_sim:    highest cosine sim between original and any proxy (best single proxy)
  - mean_sim:   average cosine sim across all selected proxies
  - coverage:   cosine sim between original embedding and centroid of proxy embeddings
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class FidelityResult:
    max_sim: float
    mean_sim: float
    coverage: float     # original vs. centroid of proxies


class BottleneckFidelityEvaluator:
    def __init__(self, embed_model: SentenceTransformer):
        self._embed_model = embed_model

    def evaluate(self, original_question: str, proxy_questions: list[str]) -> FidelityResult:
        if not proxy_questions:
            return FidelityResult(max_sim=0.0, mean_sim=0.0, coverage=0.0)

        all_texts = [original_question] + proxy_questions
        embs = self._embed_model.encode(all_texts, normalize_embeddings=True)

        orig_emb = embs[0]
        proxy_embs = embs[1:]

        sims = [float(np.dot(orig_emb, p)) for p in proxy_embs]
        centroid = proxy_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        coverage = float(np.dot(orig_emb, centroid_norm))

        return FidelityResult(
            max_sim=max(sims),
            mean_sim=float(np.mean(sims)),
            coverage=coverage,
        )
