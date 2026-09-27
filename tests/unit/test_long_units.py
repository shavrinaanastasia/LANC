import pytest
from agent_language_geometry.cli import _long_units


def test_long_units_start_at_explicit_global_index() -> None:
    cards = [{"stimulus_id": "E01"}, {"stimulus_id": "E02"}]
    units = _long_units(["competition", "neutral"], cards, [1, 2], 3)
    assert [(index, frame, card["stimulus_id"], seed) for index, frame, card, seed in units] == [
        (3, "competition", "E02", 1),
        (4, "competition", "E02", 2),
        (5, "neutral", "E01", 1),
        (6, "neutral", "E01", 2),
        (7, "neutral", "E02", 1),
        (8, "neutral", "E02", 2),
    ]


def test_long_units_reject_invalid_start_index() -> None:
    with pytest.raises(ValueError, match="between 1 and 1"):
        _long_units(["neutral"], [{"stimulus_id": "E01"}], [1], 2)
