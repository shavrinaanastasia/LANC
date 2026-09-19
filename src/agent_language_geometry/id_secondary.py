"""Secondary checks only; never substitute these estimators for PHD."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(points: np.ndarray) -> float:
    distances = NearestNeighbors(n_neighbors=3).fit(points).kneighbors(return_distance=True)[0]
    ratios = distances[:, 2] / np.maximum(distances[:, 1], np.finfo(float).eps)
    estimate = 1.0 / np.mean(np.log(ratios))
    return float(estimate) if np.isfinite(estimate) else float("nan")


def levina_bickel_mle(points: np.ndarray, k: int = 10) -> float:
    distances = (
        NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(return_distance=True)[0][:, 1:]
    )
    logs = np.log(
        np.maximum(distances[:, -1, None], np.finfo(float).eps)
        / np.maximum(distances[:, :-1], np.finfo(float).eps)
    )
    estimate = 1.0 / np.mean(logs.sum(axis=1) / (k - 1))
    return float(estimate) if np.isfinite(estimate) else float("nan")
