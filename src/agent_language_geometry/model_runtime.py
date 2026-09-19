"""Frozen SmolLM2 runtime with architecture and revision assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reproducibility import file_sha256


def load_runtime(model_config: dict[str, Any], device: str | None = None) -> tuple[Any, Any]:
    revision = model_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_config["id"], revision=revision)
    expected = model_config["expected_architecture"]
    for name, value in expected.items():
        observed = getattr(model.config, name, None)
        if observed != value:
            raise AssertionError(f"{name}={observed}, expected locked value {value}")
    if not all(not parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.eval()
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer


def model_provenance(model: Any, tokenizer: Any, revision: str) -> dict[str, Any]:
    return {
        "revision": revision,
        "config": model.config.to_dict(),
        "tokenizer_config": getattr(tokenizer, "init_kwargs", {}),
    }


def persist_model_provenance(
    directory: Path, model: Any, tokenizer: Any, revision: str
) -> dict[str, Any]:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "model_provenance.json"
    target.write_text(
        json.dumps(model_provenance(model, tokenizer, revision), sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {"path": str(target), "sha256": file_sha256(target)}
