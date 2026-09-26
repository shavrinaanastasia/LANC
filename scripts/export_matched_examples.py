"""Export prompts and matched PCA dialogues from immutable JSONL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_language_geometry.dialogue import render_frame


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def card_by_id(path: Path, stimulus_id: str) -> dict[str, Any]:
    for card in read_jsonl(path):
        if card["stimulus_id"] == stimulus_id:
            return card
    raise ValueError(f"Stimulus not found: {stimulus_id}")


def append_dialogue(lines: list[str], record: dict[str, Any]) -> None:
    lines.extend([f"## {record['intervention']}", ""])
    for utterance in record["utterances"]:
        lines.extend(
            [
                f"**{utterance['speaker']} turn {utterance['turn_index'] + 1}:** "
                f"{utterance['text']}",
                "",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--phase2-dialogues", type=Path, required=True)
    parser.add_argument("--stimulus-id", required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite example export: {args.output}")
    card = card_by_id(args.stimuli, args.stimulus_id)
    lines = [
        "# Matched Prompts And PCA Dialogues",
        "",
        f"Stimulus: `{args.stimulus_id}`. Generation seed: `{args.generation_seed}`.",
        "",
        "The six Phase 2 dialogues share this stimulus and generation seed. "
        "They differ only by intervention.",
        "",
        "# Phase 1 System Prompts",
        "",
    ]
    for frame in ("competition", "cooperation", "neutral"):
        lines.extend([f"## {frame}", ""])
        for speaker in ("A", "B"):
            lines.extend([f"### Agent {speaker}", "", render_frame(card, frame, speaker), ""])

    records = read_jsonl(args.phase2_dialogues)
    selected = {
        record["intervention"]: record
        for record in records
        if record["stimulus_id"] == args.stimulus_id
        and record["generation_seed"] == args.generation_seed
        and record["status"] == "complete"
    }
    lines.extend(["# Phase 2 Matched Dialogues", ""])
    for condition in ("identity", "pca_128", "pca_32", "pca_8", "pca_2", "pca_1"):
        if condition not in selected:
            raise ValueError(f"No complete record for condition: {condition}")
        append_dialogue(lines, selected[condition])
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
