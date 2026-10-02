"""Exact dense-MST Schweinhart estimator for explicitly bounded high-dimensional clouds."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import linregress, t


def dense_distance_gib(n_points: int) -> float:
    return n_points * n_points * np.dtype(np.float64).itemsize / 2**30


def exact_mst_energy(points: np.ndarray, alpha: float) -> float:
    distances = cdist(points, points, metric="euclidean")
    edges = minimum_spanning_tree(distances).data.astype(np.float64, copy=False)
    if len(edges) != len(points) - 1:
        raise RuntimeError("Dense Euclidean MST is disconnected")
    return float(np.power(edges, alpha).sum())


def estimate_dense_schweinhart(
    points: np.ndarray, sample_sizes: Iterable[int], alphas: Iterable[float], seed: int
) -> dict[str, object]:
    points = np.asarray(points, dtype=np.float64)
    sizes = np.asarray(sorted(set(sample_sizes)), dtype=int)
    if len(sizes) < 3 or sizes[0] < 2 or sizes[-1] > len(points):
        raise ValueError("Invalid high-dimensional MST sample-size ladder")
    permutation = np.random.default_rng(seed).permutation(len(points))
    sampled = [points[permutation[:size]] for size in sizes]
    estimates = []
    for alpha in alphas:
        energies = np.asarray([exact_mst_energy(cloud, float(alpha)) for cloud in sampled])
        fit = linregress(np.log(sizes), np.log(energies))
        estimate = float(alpha / (1.0 - fit.slope)) if fit.slope < 1.0 else None
        if fit.stderr is None or fit.slope + t.ppf(0.975, len(sizes) - 2) * fit.stderr >= 1.0:
            interval = None
        else:
            half_width = t.ppf(0.975, len(sizes) - 2) * fit.stderr
            interval = [
                float(alpha / (1.0 - (fit.slope - half_width))),
                float(alpha / (1.0 - (fit.slope + half_width))),
            ]
        estimates.append(
            {
                "alpha": float(alpha),
                "dimension": estimate,
                "dimension_ci_95": interval,
                "r_squared": float(fit.rvalue**2),
            }
        )
    return {"sample_sizes": sizes.tolist(), "alpha_estimates": estimates}
