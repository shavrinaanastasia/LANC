"""Disjoint calibration collection and float32 PCA fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from .interventions import PCABasis
from .reproducibility import file_sha256, sha256


@torch.inference_mode()
def generated_hidden_vectors(
    model: Any, tokenizer: Any, prompt_ids: torch.Tensor, output_ids: list[int]
) -> np.ndarray:
    """Re-run forced prompt+continuation and retain layer-15 outputs for generated positions only."""
    sequence = torch.cat(
        [prompt_ids[0], torch.tensor(output_ids, device=prompt_ids.device)]
    ).unsqueeze(0)
    captured: list[torch.Tensor] = []
    handle = model.model.layers[14].register_forward_hook(
        lambda _m, _a, out: captured.append((out[0] if isinstance(out, tuple) else out).detach())
    )
    try:
        model(sequence, use_cache=False)
    finally:
        handle.remove()
    hidden = captured[-1][0, prompt_ids.shape[1] :].float().cpu().numpy()
    return hidden


def deterministic_vector_sample(
    vectors: np.ndarray, maximum: int, seed: int, key: str
) -> np.ndarray:
    if len(vectors) <= maximum:
        return vectors
    local_seed = int(sha256(f"{seed}|{key}")[:16], 16) % (2**63 - 1)
    indices = np.sort(
        np.random.default_rng(local_seed).choice(len(vectors), maximum, replace=False)
    )
    return vectors[indices]


def sample_dialogue_agent_vectors(
    chunks: list[np.ndarray], maximum: int, seed: int, key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pool an agent's five turn segments before applying the preregistered cap once."""
    available = np.concatenate(chunks) if chunks else np.empty((0, 576), dtype=np.float32)
    return available, deterministic_vector_sample(available, maximum, seed, key)


def fit_pca(vectors: list[np.ndarray], artifact_path: Path) -> dict[str, Any]:
    cloud = np.concatenate(vectors).astype(np.float32, copy=False)
    if cloud.shape[1] != 576:
        raise ValueError(f"Calibration hidden dimension is {cloud.shape[1]}, expected 576")
    pca = PCA(n_components=min(cloud.shape), svd_solver="full", random_state=0).fit(cloud)
    basis = PCABasis(
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        pca.explained_variance_.astype(np.float32),
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=False)
    basis.save(artifact_path)
    return {
        "n_vectors": int(len(cloud)),
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
