"""Strict YAML configuration loading with explicit one-level inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import canonical_json, sha256


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    current = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if parent_name := current.get("extends"):
        parent = load_config(path.parent / parent_name)
        current = _merge(parent, current)
    validate_config(current)
    current["_config_path"] = str(path)
    current["_config_hash"] = sha256(
        canonical_json({k: v for k, v in current.items() if not k.startswith("_")})
    )
    return current


def validate_config(config: dict[str, Any]) -> None:
    required = ("schema_version", "model", "generation", "analysis", "paths")
    absent = [key for key in required if key not in config]
    if absent:
        raise ValueError(f"Missing required config keys: {absent}")
    model = config["model"]
    expected = {
        "hidden_size": 576,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "intermediate_size": 1536,
    }
    if model.get("expected_architecture") != expected:
        raise ValueError("Expected SmolLM2-135M architecture must remain locked")
    if model.get("intervention_layer_human") != 15 or model.get("intervention_layer_zero") != 14:
        raise ValueError("Intervention must be after human layer 15 / zero-based layer 14")
    generation = config["generation"]
    locked = {"turns": 10, "max_new_tokens": 48, "temperature": 0.7, "top_p": 0.9, "top_k": 50}
    for key, expected_value in locked.items():
        if generation.get(key) != expected_value:
            raise ValueError(f"Locked MVP generation setting {key} must be {expected_value}")
    if config["analysis"].get("n_min") != 40 or config["analysis"].get("primary_min_tokens") != 50:
        raise ValueError("Primary PHD protocol requires n_min=40 and eligibility N>=50")
