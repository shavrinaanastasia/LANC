import json
from pathlib import Path

from agent_language_geometry.cli import _cards, _validate_stimuli

ROOT = Path(__file__).resolve().parents[2]


def test_stimuli_are_disjoint_and_protocol_counts_are_locked() -> None:
    evaluation = _cards(ROOT / "stimuli/evaluation_base.jsonl")
    calibration = _cards(ROOT / "stimuli/calibration_neutral.jsonl")
    _validate_stimuli(evaluation)
    _validate_stimuli(calibration)
    assert not {card["stimulus_id"] for card in evaluation} & {
        card["stimulus_id"] for card in calibration
    }
    assert 24 * 3 * 3 == 216
    assert 24 * 6 * 3 == 432
    assert 24 * 2 == 48


def test_stimulus_jsonl_is_valid_json() -> None:
    for name in ("evaluation_base.jsonl", "calibration_neutral.jsonl"):
        for line in (ROOT / "stimuli" / name).read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["topic"]
