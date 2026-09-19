import numpy as np
import torch
from agent_language_geometry.interventions import PCABasis


def test_identity_is_bitwise_and_projection_is_nested_idempotent() -> None:
    basis = PCABasis(
        np.zeros(576, dtype="float32"), np.eye(576, dtype="float32"), np.ones(576, dtype="float32")
    )
    hidden = torch.randn(2, 3, 576)
    assert basis.project(hidden, None) is hidden
    projected = basis.project(hidden, 8)
    assert torch.allclose(projected, basis.project(projected, 8))
    assert torch.linalg.matrix_rank((projected - torch.tensor(basis.mean)).reshape(-1, 576)) <= 8
    assert torch.allclose(basis.project(hidden, 1), basis.project(basis.project(hidden, 8), 1))
