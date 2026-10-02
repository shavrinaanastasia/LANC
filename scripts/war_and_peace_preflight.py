"""Create an immutable manifest for the chapter-as-document War and Peace adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROMAN_CHAPTER = re.compile(r"(?m)^\s*[IVXLCDM]+\s*$")
RUSSIAN_WORD = re.compile(r"[А-Яа-яЁё]+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = args.text.read_text(encoding="utf-8")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.text.read_bytes()).hexdigest()
    if metadata.get("sha256") != digest:
        raise ValueError("Text SHA-256 does not match supplied metadata")
    headings = list(ROMAN_CHAPTER.finditer(text))
    chapters = [
        text[start.end() : end.start() if end else len(text)].strip()
        for start, end in zip(headings, [*headings[1:], None], strict=False)
    ]
    chapters = [chapter for chapter in chapters if chapter]
    if len(chapters) < 100:
        raise ValueError("Roman-numeral heading rule yielded fewer than 100 chapters")
    words = RUSSIAN_WORD.findall(text)
    payload = {
        "protocol": "war-and-peace-chapter-as-document-adaptation",
        "source": metadata,
        "character_count": len(text),
        "russian_word_tokens": len(words),
        "unique_surface_words": len(set(word.lower() for word in words)),
        "chapter_count": len(chapters),
        "chapter_heading_rule": ROMAN_CHAPTER.pattern,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
