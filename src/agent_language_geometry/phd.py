"""Primary zero-dimensional persistent-homology dimension (Tulchinskii et al.)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import linregress

from .schemas import PHDResult


def mst_total_length(points: np.ndarray, alpha: float = 1.0) -> float:
    if len(points) < 2:
        raise ValueError("MST requires at least two points")
    distances = cdist(points, points, metric="euclidean")
    edges = minimum_spanning_tree(distances).data.astype(np.float64, copy=False)
    return float(np.power(edges, alpha).sum())


def _sample_sizes(n: int, n_min: int, k: int) -> list[int]:
    sizes = np.linspace(n_min, n, num=k, dtype=int).tolist()
    if len(set(sizes)) != k:
        raise ValueError("PHD sample-size ladder has duplicate scales")
    return sizes


def estimate_phd(
    points: np.ndarray,
    *,
    alpha: float = 1.0,
    n_min: int = 40,
    k_sample_sizes: int = 8,
    subsets_per_size: int = 7,
    internal_seeds: Iterable[int] = (1741, 2861, 3917),
    primary_min_tokens: int = 50,
    primary_max_tokens: int = 510,
    n_full: int | None = None,
) -> PHDResult:
    """Appendix-B protocol; internal seeds stabilize one dialogue estimator, not replicates."""
    points = np.asarray(points, dtype=np.float32)
    observed_n = len(points)
    n_full = observed_n if n_full is None else n_full
    if n_full < primary_min_tokens:
        return PHDResult(
            False, None, "too_short", n_full, observed_n, False, diagnostic_flags=["N_LT_50"]
        )
    truncated = n_full > primary_max_tokens
    if truncated:
        points = points[:primary_max_tokens]  # Preregistered deterministic text-order truncation.
    n = len(points)
    sizes = _sample_sizes(n, n_min, k_sample_sizes)
    slopes: list[float] = []
    r2_values: list[float] = []
    flags: list[str] = ["TRUNCATED_TO_510"] if truncated else []
    seed_list = list(internal_seeds)
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        median_lengths: list[float] = []
        for size in sizes:
            lengths = []
            for _ in range(subsets_per_size):
                indices = rng.choice(n, size=size, replace=False)
                lengths.append(mst_total_length(points[indices], alpha))
            median_lengths.append(float(np.median(lengths)))
        if any(value <= 0 or not np.isfinite(value) for value in median_lengths):
            return PHDResult(
                False,
                None,
                "degenerate_mst",
                n_full,
                n,
                truncated,
                sample_sizes=sizes,
                internal_seeds=seed_list,
                diagnostic_flags=[*flags, "NONPOSITIVE_MST"],
            )
        regression = linregress(np.log(sizes), np.log(median_lengths))
        slopes.append(float(regression.slope))
        r2_values.append(float(regression.rvalue**2))
    kappa = float(np.mean(slopes))
    denominator = 1.0 - kappa
    if not np.isfinite(kappa) or denominator <= 0:
        return PHDResult(
            False,
            None,
            "invalid_slope",
            n_full,
            n,
            truncated,
            slopes,
            r2_values,
            sizes,
            seed_list,
            [*flags, "KAPPA_GE_1"],
        )
    value = alpha / denominator
    if not np.isfinite(value):
        return PHDResult(
            False,
            None,
            "nonfinite",
            n_full,
            n,
            truncated,
            slopes,
            r2_values,
            sizes,
            seed_list,
            flags,
        )
    return PHDResult(
        True, float(value), None, n_full, n, truncated, slopes, r2_values, sizes, seed_list, flags
    )
