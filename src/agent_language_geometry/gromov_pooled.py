"""Optional Gromov-aligned pooled analysis, deliberately gated for MVP-scale data."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .phd import mst_total_length

INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL = "INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL"


class InsufficientGromovSample(RuntimeError):
    pass


@dataclass(frozen=True)
class GromovProtocol:
    sample_size_ladder: tuple[int, ...] = (100000, 200000, 400000, 800000)
    dimensions: tuple[int, ...] = (5, 10, 15)
    alphas: tuple[float, ...] = (0.5, 1.0, 1.5)


def pooled_terms(dialogues: list[str], ngram: int = 1) -> tuple[list[str], np.ndarray]:
    """Documented minimal preprocessing: lowercase alphabetic terms, no lemmatization or stop-word removal."""
    vocabulary: dict[str, int] = {}
    rows: list[list[str]] = []
    for dialogue in dialogues:
        tokens = re.findall(r"[a-z]+", dialogue.lower())
        terms = (
            tokens
            if ngram == 1
            else [
                "_".join(tokens[index : index + ngram]) for index in range(len(tokens) - ngram + 1)
            ]
        )
        rows.append(terms)
        for term in terms:
            vocabulary.setdefault(term, len(vocabulary))
    matrix = np.zeros((len(vocabulary), len(dialogues)), dtype=np.float32)
    for column, terms in enumerate(rows):
        for term in terms:
            matrix[vocabulary[term], column] += 1
    return list(vocabulary), matrix


def svd_embeddings(word_context: np.ndarray, dimension: int) -> np.ndarray:
    if min(word_context.shape) <= dimension:
        raise InsufficientGromovSample(INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL)
    return TruncatedSVD(n_components=dimension, random_state=0).fit_transform(word_context)


def require_supported_ladder(n_unique_points: int, ladder: tuple[int, ...]) -> None:
    if len(ladder) < 2 or n_unique_points < max(ladder):
        raise InsufficientGromovSample(INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL)


def mst_growth_regression(
    points: np.ndarray, protocol: GromovProtocol | None = None, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Exploratory pooled estimate only after the predeclared scale gate is met."""
    protocol = protocol or GromovProtocol()
    require_supported_ladder(len(points), protocol.sample_size_ladder)
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}
    for alpha in protocol.alphas:
        energies = [
            mst_total_length(points[rng.choice(len(points), size, replace=False)], alpha)
            for size in protocol.sample_size_ladder
        ]
        slope, intercept = np.polyfit(np.log(protocol.sample_size_ladder), np.log(energies), 1)
        results[str(alpha)] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "dimension": float(alpha / (1 - slope)),
        }
    return results
