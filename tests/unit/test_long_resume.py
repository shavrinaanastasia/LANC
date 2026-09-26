import json

import pytest
from agent_language_geometry.cli import _read_long_record
from agent_language_geometry.reproducibility import canonical_json, sha256


def test_long_resume_rejects_tampered_record(tmp_path) -> None:
    record = {
        "run_id": "long-run",
        "protocol_metadata": {"long_config_hash": "config-hash"},
        "status": "complete",
    }
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps({**record, "record_hash": sha256(canonical_json(record))}), encoding="utf-8"
    )
    assert _read_long_record(path, "long-run", "config-hash") == record

    path.write_text(
        json.dumps({**record, "status": "tampered", "record_hash": "bad"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash"):
        _read_long_record(path, "long-run", "config-hash")
