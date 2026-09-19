"""Scriptable experimental workflow. Generated outputs are confined to immutable run directories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import deterministic_vector_sample, fit_pca, generated_hidden_vectors
from .config import load_config, validate_config
from .degeneration import degeneration_metrics
from .dialogue import FORBIDDEN_FRAME_MARKERS, generate_dialogue
from .interventions import LayerIntervention, PCABasis
from .model_runtime import load_runtime
from .phd import estimate_phd
from .reporting import render_report
from .reproducibility import (
    canonical_json,
    file_sha256,
    git_state,
    make_run_id,
    sha256,
    software_manifest,
    write_manifest,
)
from .schemas import AnalysisRecord, DialogueRecord
from .statistics import aggregate_generation_seeds, holm_adjust, paired_differences, paired_summary
from .token_embeddings import contextual_token_cloud, dialogue_analysis_text, load_analysis_encoder


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cards(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _validate_stimuli(cards: list[dict[str, Any]]) -> None:
    if len(cards) != 24 or len({card["stimulus_id"] for card in cards}) != 24:
        raise ValueError("A stimulus set must contain exactly 24 unique cards")
    strata = [card["stratum"] for card in cards]
    if any(strata.count(stratum) != 4 for stratum in set(strata)) or len(set(strata)) != 6:
        raise ValueError("Stimuli require six strata with four cards each")
    for card in cards:
        for frame, markers in FORBIDDEN_FRAME_MARKERS.items():
            rendered = (
                "".join(card.get(frame, {}).values()) if isinstance(card.get(frame), dict) else ""
            )
            if any(marker in rendered.lower() for marker in markers):
                raise ValueError(f"Frame marker leakage in {card['stimulus_id']} {frame}")


def _run_directory(config: dict[str, Any], label: str) -> Path:
    run_id = config.get("run_id") or make_run_id({**config, "label": label})
    path = _root() / config["paths"]["results_root"] / run_id
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run directory: {path}")
    path.mkdir(parents=True)
    write_manifest(
        path,
        {
            "run_id": run_id,
            "config": config,
            "config_hash": config["_config_hash"],
            "git": git_state(_root()),
        },
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def run_generation(
    config: dict[str, Any], *, limit_cards: int | None = None, limit_seeds: int | None = None
) -> Path:
    cards = _cards(_root() / config["paths"]["evaluation_stimuli"])
    _validate_stimuli(cards)
    cards = cards[:limit_cards] if limit_cards else cards
    seeds = (
        config["generation"]["seeds"][:limit_seeds]
        if limit_seeds
        else config["generation"]["seeds"]
    )
    run_dir = _run_directory(config, config.get("phase", "generation"))
    model, tokenizer = load_runtime(config["model"])
    basis = (
        PCABasis.load(_root() / config["pca_artifact"])
        if any(item != "identity" for item in config.get("interventions", []))
        else None
    )
    rows: list[dict[str, Any]] = []
    for frame in config["frames"]:
        for intervention_name in config["interventions"]:
            for card in cards:
                for seed in seeds:
                    started = datetime.now(timezone.utc).isoformat()
                    hook = LayerIntervention(intervention_name, basis).install(model)
                    try:
                        utterances = generate_dialogue(
                            model,
                            tokenizer,
                            card,
                            frame,
                            seed,
                            config["generation"],
                            debug_token_ids=False,
                        )
                        status, error = "complete", None
                    except Exception as exc:  # Persist a failed unit; never silently drop it.
                        utterances, status, error = [], "error", repr(exc)
                    finally:
                        hook.remove()
                    rows.append(
                        DialogueRecord(
                            run_dir.name,
                            config.get("phase", "generation"),
                            card["stimulus_id"],
                            sha256(canonical_json(card)),
                            frame,
                            intervention_name,
                            hook.rank,
                            seed,
                            utterances,
                            config["model"]["revision"],
                            sha256(canonical_json(config["generation"])),
                            git_state(_root()),
                            software_manifest(),
                            started,
                            datetime.now(timezone.utc).isoformat(),
                            status,
                            error,
                        ).as_dict()
                    )
    _write_jsonl(run_dir / "dialogues.jsonl", rows)
    return run_dir


def calibrate(
    config: dict[str, Any], *, limit_cards: int | None = None, limit_seeds: int | None = None
) -> Path:
    cards = _cards(_root() / config["paths"]["calibration_stimuli"])
    evaluation = _cards(_root() / config["paths"]["evaluation_stimuli"])
    _validate_stimuli(cards)
    if {card["stimulus_id"] for card in cards} & {card["stimulus_id"] for card in evaluation}:
        raise ValueError("Calibration and evaluation stimuli must be disjoint")
    cards, seeds = (
        cards[:limit_cards] if limit_cards else cards,
        config["generation"]["seeds"][:limit_seeds]
        if limit_seeds
        else config["generation"]["seeds"],
    )
    run_dir = _run_directory(config, "calibration")
    model, tokenizer = load_runtime(config["model"])
    vectors: list[Any] = []
    counts: list[dict[str, Any]] = []
    dialogue_rows: list[dict[str, Any]] = []
    calibration = config["calibration"]
    for card in cards:
        for seed in seeds:
            stimulus_id = card["stimulus_id"]

            def collect(
                prompt_ids: Any,
                ids: list[int],
                speaker: str,
                turn: int,
                *,
                stimulus_id: str = stimulus_id,
                generation_seed: int = seed,
            ) -> None:
                hidden = generated_hidden_vectors(model, tokenizer, prompt_ids, ids)
                sampled = deterministic_vector_sample(
                    hidden,
                    calibration["max_vectors_per_dialogue_per_agent"],
                    calibration["sampling_seed"],
                    f"{stimulus_id}|{generation_seed}|{speaker}|{turn}",
                )
                vectors.append(sampled)
                counts.append(
                    {
                        "stimulus_id": stimulus_id,
                        "seed": generation_seed,
                        "speaker": speaker,
                        "turn": turn,
                        "available": len(hidden),
                        "selected": len(sampled),
                    }
                )

            started = datetime.now(timezone.utc).isoformat()
            hook = LayerIntervention("identity").install(model)
            try:
                utterances = generate_dialogue(
                    model,
                    tokenizer,
                    card,
                    "neutral",
                    seed,
                    config["generation"],
                    on_generated_tokens=collect,
                )
                status, error = "complete", None
            except Exception as exc:
                utterances, status, error = [], "error", repr(exc)
            finally:
                hook.remove()
            dialogue_rows.append(
                DialogueRecord(
                    run_dir.name,
                    "calibration",
                    card["stimulus_id"],
                    sha256(canonical_json(card)),
                    "neutral",
                    "identity",
                    None,
                    seed,
                    utterances,
                    config["model"]["revision"],
                    sha256(canonical_json(config["generation"])),
                    git_state(_root()),
                    software_manifest(),
                    started,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    error,
                ).as_dict()
            )
    _write_jsonl(run_dir / "calibration_dialogues.jsonl", dialogue_rows)
    if not vectors:
        raise RuntimeError("Calibration produced no generated hidden vectors")
    artifact_dir = _root() / config["paths"]["calibration_root"] / run_dir.name
    artifact = artifact_dir / "pca_basis.npz"
    info = fit_pca(vectors, artifact)
    (run_dir / "calibration_manifest.json").write_text(
        canonical_json(
            {
                "counts": counts,
                **info,
                "model_revision": config["model"]["revision"],
                "stimuli_hash": file_sha256(_root() / config["paths"]["calibration_stimuli"]),
                "config_hash": config["_config_hash"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def analyze(run_dir: Path, config: dict[str, Any]) -> Path:
    dialogue_rows = [
        json.loads(line)
        for line in (run_dir / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    model, tokenizer = load_analysis_encoder(config["analysis"])
    rows: list[dict[str, Any]] = []
    for dialogue in dialogue_rows:
        utterances = dialogue["utterances"]
        text = dialogue_analysis_text(utterances)
        points, n_full, _ = (
            contextual_token_cloud(model, tokenizer, text)
            if text
            else (__import__("numpy").empty((0, 768), dtype="float32"), 0, False)
        )
        phd = estimate_phd(
            points,
            n_full=n_full,
            **{
                key: config["analysis"][key]
                for key in (
                    "alpha",
                    "n_min",
                    "k_sample_sizes",
                    "subsets_per_size",
                    "internal_seeds",
                    "primary_min_tokens",
                    "primary_max_tokens",
                )
            },
        )
        metrics = degeneration_metrics(
            utterances, roberta_content_tokens=phd.n_full, truncated=phd.truncated
        )
        condition = (
            dialogue["interaction_frame"]
            if dialogue["phase"] in {"phase1", "smoke"}
            else dialogue["intervention"]
        )
        rows.append(
            AnalysisRecord(
                dialogue["run_id"],
                dialogue["stimulus_id"],
                condition,
                dialogue["generation_seed"],
                config["analysis"]["encoder_id"],
                config["analysis"]["encoder_revision"],
                config["_config_hash"],
                phd,
                metrics,
            ).as_dict()
        )
    _write_jsonl(run_dir / "analysis.jsonl", rows)
    phase = dialogue_rows[0]["phase"] if dialogue_rows else ""
    contrasts = (
        (("competition", "neutral"), ("cooperation", "neutral"), ("competition", "cooperation"))
        if phase in {"phase1", "smoke"}
        else (
            ("pca_128", "identity"),
            ("pca_32", "identity"),
            ("pca_8", "identity"),
            ("pca_2", "identity"),
            ("pca_1", "identity"),
        )
    )
    aggregated = aggregate_generation_seeds(rows)
    summaries = {
        f"{left}-{right}": paired_summary(
            paired_differences(aggregated, left, right),
            bootstrap_iterations=config["bootstrap_iterations"],
            permutation_iterations=config["permutation_iterations"],
        )
        for left, right in contrasts
    }
    adjusted = holm_adjust(
        {name: float(summary.get("sign_flip_p", 1.0)) for name, summary in summaries.items()}
    )
    (run_dir / "statistics.json").write_text(
        canonical_json({"phase": phase, "contrasts": summaries, "holm_adjusted_p": adjusted})
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-language-geometry")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "smoke", "calibrate-pca", "run-phase1", "run-phase2"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "run-phase2":
            command.add_argument("--pca-artifact")
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--config", required=True)
    analyze_parser.add_argument("--run-dir", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "report":
        print(render_report(Path(args.run_dir)))
        return
    config = load_config(args.config)
    if args.command == "validate-config":
        validate_config(config)
        print(f"valid {config['_config_hash']}")
    elif args.command == "calibrate-pca":
        print(calibrate(config))
    elif args.command == "analyze":
        print(analyze(Path(args.run_dir), config))
    elif args.command == "smoke":
        smoke_config = {
            **config,
            "phase": "smoke",
            "frames": config.get("frames", ["competition", "cooperation", "neutral"]),
            "interventions": config.get("interventions", ["identity"]),
        }
        print(run_generation(smoke_config, limit_cards=2, limit_seeds=1))
    else:
        if args.command == "run-phase2" and args.pca_artifact:
            config = {**config, "pca_artifact": args.pca_artifact}
        print(run_generation(config))


if __name__ == "__main__":
    main()
