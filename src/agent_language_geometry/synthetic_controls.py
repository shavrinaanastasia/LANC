"""Synthetic PHD controls: diagnostics, never experimental language data."""

from __future__ import annotations

import numpy as np

from .phd import estimate_phd


def control_clouds(n: int = 100, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    swiss_t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    return {
        "identical": np.zeros((n, 3), dtype=np.float32),
        "gaussian_2": rng.normal(size=(n, 2)).astype(np.float32),
        "gaussian_8": rng.normal(size=(n, 8)).astype(np.float32),
        "noisy_line": np.column_stack(
            (np.linspace(-1, 1, n), rng.normal(scale=0.02, size=n))
        ).astype(np.float32),
        "noisy_circle": np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
        + rng.normal(scale=0.02, size=(n, 2)),
        "swiss_roll": np.column_stack(
            (swiss_t * np.cos(swiss_t), rng.random(n), swiss_t * np.sin(swiss_t))
        ).astype(np.float32),
    }


def run_controls(n: int = 100, seed: int = 0) -> dict[str, object]:
    return {
        name: estimate_phd(points).as_dict() for name, points in control_clouds(n, seed).items()
    }
