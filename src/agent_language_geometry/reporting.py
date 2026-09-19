"""Reports are generated only from persisted tidy JSONL tables."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

TEN_POINT_EXPLANATION = "A ten-utterance dialogue is not a ten-point ID sample: PHD uses contextual token embeddings (50-510 points) inside one dialogue. Generation seeds are technical repeats aggregated within stimulus; only stimulus blocks support primary inference."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_report(run_dir: Path) -> Path:
    analysis_path = run_dir / "analysis.jsonl"
    rows = read_jsonl(analysis_path) if analysis_path.exists() else []
    condition_counts = Counter(row["condition"] for row in rows)
    invalid = Counter(row["condition"] for row in rows if not row["phd"]["valid"])
    flags = Counter(flag for row in rows for flag in row["phd"].get("diagnostic_flags", []))
    lines = [
        "# Language Geometry Report",
        "",
        "## Protocol",
        "",
        TEN_POINT_EXPLANATION,
        "",
        "Gromov-style pooled word/n-gram analysis is separate from dialogue PHD and is not reported unless its predefined ladder is supported.",
        "",
        "## Degeneration Panel",
        "",
    ]
    for condition, count in sorted(condition_counts.items()):
        rate = invalid[condition] / count if count else 0
        label = " DEGENERATED_OR_UNSTABLE" if rate > 0.20 else ""
        lines.append(
            f"- `{condition}`: {count} dialogues, invalid PHD {invalid[condition]}/{count} ({rate:.1%}){label}"
        )
    lines.extend(
        [
            "",
            "## PHD Diagnostics",
            "",
            f"- Flags: `{dict(flags)}`",
            "- Primary PHD is shown only after the degeneration panel; invalid results are never imputed.",
        ]
    )
    target = run_dir / "report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
