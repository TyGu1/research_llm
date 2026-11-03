"""
Batched answer-quality scoring for the adaptive-attack evaluation.

Same metrics as AnswerQualityEvaluator (BERTScore F1 vs oracle, semantic similarity
vs oracle, gold-span overlap, length ratio) but the BERTScore model is loaded once
and all pairs are scored in batches — the original evaluator reloads roberta-large
for every pair.
"""
from __future__ import annotations

import numpy as np


def gold_span_overlap(prediction: str, gold: list[str]) -> float:
    if not gold:
        return 0.0
    p = prediction.lower()
    return sum(1 for a in gold if a.lower()[:50] in p) / len(gold)


class BatchedQualityScorer:
    def __init__(self, embed_model, bertscore_model: str = "roberta-large", device: str = "cpu", batch_size: int = 16):
        from bert_score import BERTScorer
        self._embed = embed_model
        self._bs = BERTScorer(model_type=bertscore_model, device=device, batch_size=batch_size, lang="en")

    def score(self, predictions: list[str], oracles: list[str], golds: list[list[str]], max_chars: int = 8000) -> list[dict]:
        pairs = [(i, p[:max_chars], o[:max_chars]) for i, (p, o) in enumerate(zip(predictions, oracles)) if p and o]
        f1 = {}
        if pairs:
            _, _, F = self._bs.score([p for _, p, _ in pairs], [o for _, _, o in pairs])
            for (i, _, _), v in zip(pairs, F.tolist()):
                f1[i] = float(v)
        sem = {}
        if pairs:
            embs_p = self._embed.encode([p for _, p, _ in pairs], normalize_embeddings=True, batch_size=32)
            embs_o = self._embed.encode([o for _, _, o in pairs], normalize_embeddings=True, batch_size=32)
            for (i, _, _), a, b in zip(pairs, embs_p, embs_o):
                sem[i] = float(np.dot(a, b))
        out = []
        for i, (p, o, g) in enumerate(zip(predictions, oracles, golds)):
            out.append({
                "bertscore_f1": f1.get(i, 0.0),
                "semantic_sim": sem.get(i, 0.0),
                "gold_span_overlap": gold_span_overlap(p or "", g),
                "length_ratio": (len(p) / len(o)) if (p and o) else 0.0,
            })
        return out
