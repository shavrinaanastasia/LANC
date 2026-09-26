"""Synthetic controls for the Schweinhart MST-growth estimator.

These controls validate numerical behaviour on point clouds with known
Hausdorff dimensions. They are not language data and do not alter dialogue PHD.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.stats import linregress, t

KNOWN_DIMENSIONS = {
    "sphere": 2.0,
    "swiss_roll": 2.0,
    "sierpinski_carpet": float(np.log(8) / np.log(3)),
    "menger_sponge": float(np.log(20) / np.log(3)),
}


def control_clouds(n: int, seed: int) -> dict[str, np.ndarray]:
    """Return deterministic samples from the four Appendix-A control shapes."""
    if n < 4:
        raise ValueError("Synthetic controls require at least four points")
    rng = np.random.default_rng(seed)

    sphere = rng.normal(size=(n, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)

    roll_t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    swiss_roll = np.column_stack(
        (roll_t * np.cos(roll_t), rng.uniform(-1, 1, size=n), roll_t * np.sin(roll_t))
    )

    carpet = rng.random((n, 2))
    for _ in range(20):
        digits = rng.integers(0, 3, size=(n, 2))
        while np.any(np.all(digits == 1, axis=1)):
            invalid = np.all(digits == 1, axis=1)
            digits[invalid] = rng.integers(0, 3, size=(invalid.sum(), 2))
        carpet = (carpet + digits) / 3.0

    shifts = np.array(
        [item for item in np.ndindex(3, 3, 3) if sum(coordinate == 1 for coordinate in item) < 2],
        dtype=np.float64,
    )
    sponge = rng.random((n, 3))
    for _ in range(20):
        sponge = (sponge + shifts[rng.integers(0, len(shifts), size=n)]) / 3.0

    return {
        "sphere": sphere,
        "swiss_roll": swiss_roll,
        "sierpinski_carpet": carpet,
        "menger_sponge": sponge,
    }


def mst_edge_lengths(points: np.ndarray) -> np.ndarray:
    """Compute the exact Euclidean MST from Delaunay candidate edges in 2D or 3D."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("Delaunay MST controls support only two- and three-dimensional clouds")
    n = len(points)
    simplices = Delaunay(points, qhull_options="QJ Qbb Qc").simplices
    candidate_edges = np.concatenate(
        [simplices[:, pair] for pair in combinations(range(points.shape[1] + 1), 2)]
    )
    candidate_edges.sort(axis=1)
    candidate_edges = np.unique(candidate_edges, axis=0)
    lengths = np.linalg.norm(points[candidate_edges[:, 0]] - points[candidate_edges[:, 1]], axis=1)
    graph = coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((candidate_edges[:, 0], candidate_edges[:, 1])),
                np.concatenate((candidate_edges[:, 1], candidate_edges[:, 0])),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    edges = minimum_spanning_tree(graph).data
    if len(edges) != n - 1:
        raise RuntimeError("Delaunay candidate graph did not yield a spanning tree")
    return edges.astype(np.float64, copy=False)


def _estimate_for_alpha(
    edge_lengths: list[np.ndarray], sample_sizes: np.ndarray, alpha: float
) -> dict[str, float | list[float] | None]:
    energies = np.asarray([np.power(edges, alpha).sum() for edges in edge_lengths])
    fit = linregress(np.log(sample_sizes), np.log(energies))
    estimate = alpha / (1.0 - fit.slope) if fit.slope < 1.0 else None
    if fit.stderr is None or len(sample_sizes) <= 2:
        interval = None
    else:
        half_width = t.ppf(0.975, len(sample_sizes) - 2) * fit.stderr
        lower_slope, upper_slope = fit.slope - half_width, fit.slope + half_width
        interval = (
            [alpha / (1.0 - lower_slope), alpha / (1.0 - upper_slope)]
            if upper_slope < 1.0
            else None
        )
    return {
        "alpha": alpha,
        "slope": float(fit.slope),
        "r_squared": float(fit.rvalue**2),
        "dimension": None if estimate is None else float(estimate),
        "dimension_ci_95": interval,
    }


def estimate_schweinhart(
    points: np.ndarray, sample_sizes: Iterable[int], alphas: Iterable[float], seed: int
) -> dict[str, object]:
    """Estimate dimensions from nested random samples and an alpha sweep."""
    sample_sizes_array = np.asarray(sorted(set(sample_sizes)), dtype=int)
    if len(sample_sizes_array) < 3:
        raise ValueError("Schweinhart regression requires at least three sample sizes")
    if sample_sizes_array[0] < 4 or sample_sizes_array[-1] > len(points):
        raise ValueError("Sample sizes must lie between four and the cloud size")

    indices = np.random.default_rng(seed).permutation(len(points))
    edge_lengths = [mst_edge_lengths(points[indices[:size]]) for size in sample_sizes_array]
    return {
        "sample_sizes": sample_sizes_array.tolist(),
        "alpha_estimates": [
            _estimate_for_alpha(edge_lengths, sample_sizes_array, float(alpha)) for alpha in alphas
        ],
    }


def run_schweinhart_controls(
    n: int, sample_sizes: Iterable[int], alphas: Iterable[float], seed: int
) -> dict[str, object]:
    """Run deterministic known-dimension controls with independent shape streams."""
    return {
        name: {
            "known_dimension": KNOWN_DIMENSIONS[name],
            **estimate_schweinhart(points, sample_sizes, alphas, seed + index + 1),
        }
        for index, (name, points) in enumerate(control_clouds(n, seed).items())
    }
