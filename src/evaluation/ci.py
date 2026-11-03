"""
Cluster-bootstrap confidence intervals.

CUAD questions from the same contract share parties, dates and vocabulary, so a
per-question bootstrap understates the variance. Every interval reported in the
adaptive-attack evaluation resamples CONTRACTS (clusters) with replacement and
recomputes the pooled mean over all questions of the resampled contracts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int
    n_clusters: int

    def fmt(self, pct: bool = True, digits: int = 1) -> str:
        if pct:
            return f"{100*self.mean:.{digits}f}% [{100*self.lo:.{digits}f}, {100*self.hi:.{digits}f}]"
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Percentile cluster bootstrap of the pooled mean.

    values[i] belongs to cluster clusters[i]. Clusters are resampled with
    replacement; the statistic is the mean over all values in the resampled
    clusters (so larger clusters carry proportionally more weight, exactly as in
    the point estimate).
    """
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length")
    if not values:
        return BootstrapCI(mean=0.0, lo=0.0, hi=0.0, n=0, n_clusters=0)

    by_cluster: dict[Hashable, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        by_cluster[c].append(float(v))
    keys = list(by_cluster)
    sums = np.array([sum(by_cluster[k]) for k in keys])
    counts = np.array([len(by_cluster[k]) for k in keys], dtype=float)
    point = float(sums.sum() / counts.sum())

    if len(keys) < 2:
        return BootstrapCI(mean=point, lo=point, hi=point, n=len(values), n_clusters=len(keys))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return BootstrapCI(mean=point, lo=lo, hi=hi, n=len(values), n_clusters=len(keys))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no clustering)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))
