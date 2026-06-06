"""
Test 4: Task-Type Sensitivity Analysis.

Groups evaluation results by task type and computes per-group statistics.
Tests the hypothesis that framework_reasoning/compliance_regulatory degrade minimally
while entity_specific degrades significantly through the bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from src.data.cuad_loader import CUADSample
from src.pipeline.base import PipelineResult
from src.evaluation.answer_quality import QualityResult
from src.data.question_taxonomy import TASK_TYPES


@dataclass
class TaskGroupStats:
    task_type: str
    n: int
    mean_bertscore_f1: float
    mean_semantic_sim: float
    mean_gold_span_overlap: float
    mean_specific_value_recovery: float     # from recovery test


@dataclass
class TaskSensitivityResult:
    groups: dict[str, TaskGroupStats] = field(default_factory=dict)


class TaskSensitivityEvaluator:
    def evaluate(
        self,
        samples: list[CUADSample],
        quality_results: list[QualityResult],
        specific_value_rates: list[float],
    ) -> TaskSensitivityResult:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for sample, quality, svr in zip(samples, quality_results, specific_value_rates):
            tt = sample.task_type
            grouped[tt]["bertscore_f1"].append(quality.bertscore_f1)
            grouped[tt]["semantic_sim"].append(quality.semantic_sim_vs_oracle)
            grouped[tt]["gold_span_overlap"].append(quality.gold_span_overlap)
            grouped[tt]["specific_value_rate"].append(svr)

        result = TaskSensitivityResult()
        for tt in TASK_TYPES:
            g = grouped.get(tt, {})
            if not g:
                continue
            result.groups[tt] = TaskGroupStats(
                task_type=tt,
                n=len(g["bertscore_f1"]),
                mean_bertscore_f1=float(np.mean(g["bertscore_f1"])),
                mean_semantic_sim=float(np.mean(g["semantic_sim"])),
                mean_gold_span_overlap=float(np.mean(g["gold_span_overlap"])),
                mean_specific_value_recovery=float(np.mean(g["specific_value_rate"])),
            )
        return result
