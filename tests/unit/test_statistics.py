import numpy as np
from agent_language_geometry.statistics import holm_adjust, paired_summary


def test_paired_summary_and_holm() -> None:
    result = paired_summary(
        np.array([1.0, 2.0, 3.0]), bootstrap_iterations=100, permutation_iterations=100
    )
    assert result["valid_paired_blocks"] == 3
    assert result["paired_mean_difference"] == 2.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
