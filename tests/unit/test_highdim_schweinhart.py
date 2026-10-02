import numpy as np
from agent_language_geometry.highdim_schweinhart import estimate_dense_schweinhart


def test_dense_highdim_estimator_is_reproducible_and_finite() -> None:
    points = np.random.default_rng(17).uniform(size=(400, 5))
    first = estimate_dense_schweinhart(points, (100, 200, 400), (1.0,), 91)
    assert first == estimate_dense_schweinhart(points, (100, 200, 400), (1.0,), 91)
    assert first["alpha_estimates"][0]["dimension"] is not None
