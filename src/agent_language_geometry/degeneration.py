"""Language-existence controls reported alongside, and before, phase-2 PHD."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
META = re.compile(r"\b(prompt|instruction|system message|as an ai|language model)\b", re.I)
REFUSAL = re.compile(r"\b(I cannot|I can't|unable to|won't)\b", re.I)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]


def _distinct(tokens: list[str], n: int) -> float:
    grams = _ngrams(tokens, n)
    return len(set(grams)) / len(grams) if grams else 0.0


def _mattr(tokens: list[str], window: int = 25) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(
        np.mean(
            [len(set(tokens[i : i + window])) / window for i in range(len(tokens) - window + 1)]
        )
    )


def _max_repeated_substring(text: str) -> int:
    words = text.lower().split()
    longest = 0
    for width in range(1, len(words) // 2 + 1):
        seen: set[tuple[str, ...]] = set()
        for start in range(len(words) - width + 1):
            phrase = tuple(words[start : start + width])
            if phrase in seen:
                longest = width
            seen.add(phrase)
    return longest


def degeneration_metrics(
    utterances: list[dict[str, Any]] | list[Any],
    *,
    model_tokens: int | None = None,
    roberta_content_tokens: int | None = None,
    truncated: bool = False,
    embed: Callable[[list[str]], np.ndarray] | None = None,
    stimulus_text: str | None = None,
) -> dict[str, Any]:
    texts = [item["text"] if isinstance(item, dict) else item.text for item in utterances]
    text = "\n".join(texts)
    tokens = [word.lower() for word in WORD.findall(text)]
    by_speaker = {
        speaker: [len(WORD.findall(texts[index])) for index in range(speaker_index, len(texts), 2)]
        for speaker, speaker_index in (("A", 0), ("B", 1))
    }
    three, four = _ngrams(tokens, 3), _ngrams(tokens, 4)
    utterance_tokens = [[word.lower() for word in WORD.findall(value)] for value in texts]
    copy_pairs = [
        (left, right)
        for left in range(len(utterance_tokens))
        for right in range(left + 1, len(utterance_tokens))
    ]

    def copy_rate(pairs: list[tuple[int, int]]) -> float:
        return (
            float(
                np.mean(
                    [
                        len(set(utterance_tokens[left]) & set(utterance_tokens[right]))
                        / max(1, len(set(utterance_tokens[right])))
                        for left, right in pairs
                    ]
                )
            )
            if pairs
            else 0.0
        )

    metrics: dict[str, Any] = {
        "empty": not bool(tokens),
        "too_short": roberta_content_tokens is not None and roberta_content_tokens < 50,
        "truncated": truncated,
        "total_words": len(tokens),
        "model_tokens": model_tokens,
        "roberta_content_tokens": roberta_content_tokens,
        "utterance_word_lengths": by_speaker,
        "early_eos_rate": float(sum(not item for item in texts) / len(texts)) if texts else 1.0,
        "repeated_3gram_fraction": 1 - len(set(three)) / len(three) if three else 0.0,
        "repeated_4gram_fraction": 1 - len(set(four)) / len(four) if four else 0.0,
        "distinct_1": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "distinct_2": _distinct(tokens, 2),
        "type_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "mattr": _mattr(tokens),
        "non_alphabetic_token_fraction": float(
            sum(not token.isalpha() for token in text.split()) / len(text.split())
        )
        if text.split()
        else 0.0,
        "max_repeated_substring_words": _max_repeated_substring(text),
        "self_copy_rate": copy_rate(
            [(left, right) for left, right in copy_pairs if left % 2 == right % 2]
        ),
        "cross_speaker_copy_rate": copy_rate(
            [(left, right) for left, right in copy_pairs if left % 2 != right % 2]
        ),
        "refusal_meta_marker_rate": float(
            sum(bool(META.search(value) or REFUSAL.search(value)) for value in texts) / len(texts)
        )
        if texts
        else 0.0,
    }
    if embed is not None and texts:
        vectors = embed(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=np.finfo(float).eps)
        vectors = vectors / norms
        metrics["adjacent_utterance_cosine_similarity"] = (
            float(np.mean((vectors[:-1] * vectors[1:]).sum(1))) if len(vectors) > 1 else None
        )
        if stimulus_text:
            stimulus = embed([stimulus_text])[0]
            metrics["stimulus_dialogue_semantic_similarity"] = float(
                (vectors.mean(0) @ stimulus)
                / (np.linalg.norm(vectors.mean(0)) * np.linalg.norm(stimulus))
            )
    return metrics
