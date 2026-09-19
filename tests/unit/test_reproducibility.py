from agent_language_geometry.reproducibility import derive_turn_seed


def test_turn_seed_is_stable_and_condition_free() -> None:
    first = derive_turn_seed(1103, "E01", 0, "A")
    assert first == derive_turn_seed(1103, "E01", 0, "A")
    assert first != derive_turn_seed(1103, "E01", 1, "A")
    assert first != derive_turn_seed(1103, "E01", 0, "B")
