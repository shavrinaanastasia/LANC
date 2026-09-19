import numpy as np
from agent_language_geometry.phd import estimate_phd, mst_total_length


def test_phd_is_reproducible_on_fixed_cloud() -> None:
    cloud = np.random.default_rng(4).normal(size=(80, 4)).astype("float32")
    first = estimate_phd(cloud)
    second = estimate_phd(cloud)
    assert first.valid and first.value is not None
    assert first == second


def test_phd_rejects_short_cloud() -> None:
    result = estimate_phd(np.zeros((49, 3), dtype="float32"))
    assert not result.valid
    assert result.reason == "too_short"


def test_mst_matches_triangle_reference() -> None:
    points = np.array([[0, 0], [3, 0], [0, 4]], dtype="float32")
    assert mst_total_length(points) == 7.0
