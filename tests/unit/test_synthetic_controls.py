from agent_language_geometry.synthetic_controls import control_clouds, run_controls


def test_synthetic_controls_are_reproducible_and_finite_when_valid() -> None:
    first = run_controls(80, 7)
    assert first == run_controls(80, 7)
    assert not first["identical"]["valid"]
    assert first["gaussian_2"]["valid"]
    assert first["gaussian_8"]["valid"]
    assert first["gaussian_2"]["value"] < first["gaussian_8"]["value"]
    assert control_clouds(50, 1)["swiss_roll"].shape == (50, 3)
