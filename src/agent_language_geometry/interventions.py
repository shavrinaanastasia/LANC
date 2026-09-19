"""Layer-15 (one-based) identity and nested PCA interventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class PCABasis:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != (576,) or self.components.shape[1] != 576:
            raise ValueError("PCA basis must operate on the locked 576-dimensional hidden state")

    @classmethod
    def load(cls, path: str | Path) -> PCABasis:
        artifact = np.load(path)
        return cls(artifact["mean"], artifact["components"], artifact["explained_variance"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            explained_variance=self.explained_variance,
        )

    def project(self, hidden: torch.Tensor, rank: int | None) -> torch.Tensor:
        if rank is None:
            return hidden
        if not 1 <= rank <= self.components.shape[0]:
            raise ValueError(f"Invalid PCA rank: {rank}")
        mean = torch.as_tensor(self.mean, dtype=hidden.dtype, device=hidden.device)
        components = torch.as_tensor(
            self.components[:rank], dtype=hidden.dtype, device=hidden.device
        )
        centered = hidden - mean
        # rows of components are orthonormal PCA axes: U_k @ U_k.T (h - mu)
        return mean + (centered @ components.T) @ components


def intervention_rank(name: str) -> int | None:
    if name == "identity":
        return None
    if name.startswith("pca_") and name[4:].isdigit():
        return int(name[4:])
    raise ValueError(f"Unknown intervention: {name}")


class LayerIntervention:
    """Forward hook at `model.layers[14]`, applied on all prefill and decode calls."""

    def __init__(self, condition: str, basis: PCABasis | None = None) -> None:
        self.condition = condition
        self.rank = intervention_rank(condition)
        if self.rank is not None and basis is None:
            raise ValueError("PCA intervention requires a calibration basis")
        self.basis = basis
        self.calls = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def transform(self, output: object) -> object:
        self.calls += 1
        if self.rank is None:
            return output  # Preserve object and tensor bitwise for identity.
        if isinstance(output, tuple):
            hidden, *remainder = output
            return (self.basis.project(hidden, self.rank), *remainder)  # type: ignore[union-attr]
        # Modern Transformers layers return BaseModelOutput-like objects only rarely here.
        if hasattr(output, "hidden_states"):
            output.hidden_states = self.basis.project(output.hidden_states, self.rank)  # type: ignore[union-attr]
            return output
        return self.basis.project(output, self.rank)  # type: ignore[arg-type,union-attr]

    def install(self, model: torch.nn.Module) -> LayerIntervention:
        try:
            layer = model.model.layers[14]
        except AttributeError as exc:
            raise RuntimeError(
                "Expected Llama model.model.layers[14] boundary is unavailable"
            ) from exc
        self._handle = layer.register_forward_hook(
            lambda _module, _args, output: self.transform(output)
        )
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> LayerIntervention:
        return self

    def __exit__(self, *_: object) -> None:
        self.remove()
