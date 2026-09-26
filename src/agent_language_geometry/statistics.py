"""Stimulus-blocked paired inference; generation seeds are aggregated first."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def aggregate_generation_seeds(rows: Iterable[dict]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        phd = row.get("phd", {})
        if phd.get("valid"):
            grouped.setdefault((row["stimulus_id"], row["condition"]), []).append(
                float(phd["value"])
            )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_differences(
    aggregated: dict[tuple[str, str], float], left: str, right: str
) -> np.ndarray:
    ids = sorted(
        {stimulus for stimulus, condition in aggregated if condition == left}
        & {stimulus for stimulus, condition in aggregated if condition == right}
    )
    return np.array(
        [aggregated[(item, left)] - aggregated[(item, right)] for item in ids], dtype=float
    )


def _bootstrap_ci(
    differences: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    means = np.empty(iterations)
    for index in range(iterations):
        means[index] = rng.choice(differences, len(differences), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())  # type: ignore[return-value]


def _sign_flip_pvalue(differences: np.ndarray, iterations: int, rng: np.random.Generator) -> float:
    observed = abs(differences.mean())
    signed_means = (
        (rng.integers(0, 2, size=(iterations, len(differences))) * 2 - 1) * differences
    ).mean(axis=1)
    null = np.abs(signed_means)
    return float((1 + np.count_nonzero(null >= observed)) / (iterations + 1))


def paired_summary(
    differences: np.ndarray,
    *,
    bootstrap_iterations: int = 10000,
    permutation_iterations: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int | list[float]]:
    if not len(differences):
        return {"valid_paired_blocks": 0}
    rng = np.random.default_rng(seed)
    standard_deviation = (
        float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    )
    return {
        "paired_mean_difference": float(differences.mean()),
        "paired_median_difference": float(np.median(differences)),
        "bootstrap_ci_95": list(_bootstrap_ci(differences, bootstrap_iterations, rng)),
        "sign_flip_p": _sign_flip_pvalue(differences, permutation_iterations, rng),
        "paired_standardized_effect_dz": float(differences.mean() / standard_deviation)
        if standard_deviation
        else float("nan"),
        "valid_paired_blocks": int(len(differences)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted
