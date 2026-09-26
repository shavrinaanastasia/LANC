"""Versioned records make experimental levels and invalid outcomes explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Utterance:
    speaker: str
    turn_index: int
    text: str
    output_token_ids: list[int] | None = None


@dataclass(frozen=True)
class DialogueRecord:
    run_id: str
    phase: str
    stimulus_id: str
    stimulus_hash: str
    interaction_frame: str
    intervention: str
    pca_rank: int | None
    generation_seed: int
    utterances: list[Utterance]
    model_revision: str
    generation_config_hash: str
    code: dict[str, Any]
    hardware_software: dict[str, Any]
    started_at: str
    ended_at: str
    status: str
    error: str | None = None
    shared_weights: bool = True
    protocol_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PHDResult:
    valid: bool
    value: float | None
    reason: str | None
    n_full: int
    n_used: int
    truncated: bool
    slopes: list[float] = field(default_factory=list)
    r2: list[float] = field(default_factory=list)
    sample_sizes: list[int] = field(default_factory=list)
    internal_seeds: list[int] = field(default_factory=list)
    diagnostic_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisRecord:
    run_id: str
    stimulus_id: str
    condition: str
    generation_seed: int
    encoder_id: str
    encoder_revision: str
    analysis_config_hash: str
    phd: PHDResult
    degeneration: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
