"""
Test 2: Answer Quality Evaluation.

Compares pipeline answer against:
  - Oracle (cloud LLM with full S2 context) — quality ceiling
  - Local-only (local LLM with S2 context) — quality floor
  - Entity substitution baseline

Metrics:
  - BERTScore F1 vs. oracle
  - Semantic similarity vs. oracle (sentence-transformers)
  - Gold span overlap (fraction of CUAD answer spans found in prediction)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer


@dataclass
class QualityResult:
    bertscore_f1: float
    semantic_sim_vs_oracle: float
    gold_span_overlap: float
    answer_length_ratio: float      # vs. oracle length (1.0 = same length)


class AnswerQualityEvaluator:
    def __init__(
        self,
        embed_model: SentenceTransformer,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        device: str = "cpu",
    ):
        self._embed_model = embed_model
        self._bertscore_model = bertscore_model
        self._device = device

    def _semantic_sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        embs = self._embed_model.encode([a, b])
        return float(
            np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
        )

    def _gold_span_overlap(self, prediction: str, gold_answers: list[str]) -> float:
        if not gold_answers:
            return 0.0
        found = sum(1 for ans in gold_answers if ans.lower()[:50] in prediction.lower())
        return found / len(gold_answers)

    def evaluate(
        self,
        prediction: str,
        oracle_answer: str,
        gold_answers: list[str],
    ) -> QualityResult:
        # BERTScore (returns tensors; take mean F1)
        if prediction and oracle_answer:
            _, _, f1 = bertscore(
                [prediction],
                [oracle_answer],
                model_type=self._bertscore_model,
                device=self._device,
                verbose=False,
            )
            bs_f1 = float(f1.mean())
        else:
            bs_f1 = 0.0

        sem_sim = self._semantic_sim(prediction, oracle_answer)
        gold_overlap = self._gold_span_overlap(prediction, gold_answers)
        length_ratio = (len(prediction) / len(oracle_answer)) if oracle_answer else 0.0

        return QualityResult(
            bertscore_f1=bs_f1,
            semantic_sim_vs_oracle=sem_sim,
            gold_span_overlap=gold_overlap,
            answer_length_ratio=length_ratio,
        )
