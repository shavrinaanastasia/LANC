from agent_language_geometry.schweinhart_controls import KNOWN_DIMENSIONS, run_schweinhart_controls


def test_schweinhart_controls_are_reproducible_and_recover_known_dimensions() -> None:
    result = run_schweinhart_controls(512, (64, 128, 256, 512), (1.0,), 91)
    assert result == run_schweinhart_controls(512, (64, 128, 256, 512), (1.0,), 91)

    for name, known_dimension in KNOWN_DIMENSIONS.items():
        estimate = result[name]["alpha_estimates"][0]["dimension"]
        assert estimate is not None
        assert abs(estimate - known_dimension) < 0.8
