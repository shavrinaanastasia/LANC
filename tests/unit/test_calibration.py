import numpy as np
from agent_language_geometry.calibration import sample_dialogue_agent_vectors


def test_dialogue_agent_cap_is_applied_after_turn_pooling() -> None:
    chunks = [np.full((48, 576), turn, dtype=np.float32) for turn in range(5)]
    available, sampled = sample_dialogue_agent_vectors(chunks, 128, 99173, "E01|1103|A")
    assert available.shape == (240, 576)
    assert sampled.shape == (128, 576)
    _, repeated = sample_dialogue_agent_vectors(chunks, 128, 99173, "E01|1103|A")
    assert np.array_equal(sampled, repeated)
