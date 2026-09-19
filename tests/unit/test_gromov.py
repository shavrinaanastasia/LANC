import pytest
from agent_language_geometry.gromov_pooled import (
    INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL,
    InsufficientGromovSample,
    require_supported_ladder,
)


def test_gromov_hard_gate() -> None:
    with pytest.raises(InsufficientGromovSample, match=INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL):
        require_supported_ladder(99999, (100000, 200000))
