"""Deterministic two-logical-agent dialogue protocol with separate histories/RNGs."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .reproducibility import derive_turn_seed
from .schemas import Utterance

FORBIDDEN_FRAME_MARKERS = {
    "competition": ("shared goal", "private clue"),
    "cooperation": ("outscore", "personal payoff"),
    "neutral": ("outscore", "personal payoff", "shared goal", "private clue"),
}


@dataclass
class LogicalAgent:
    name: str
    system_prompt: str
    history: list[dict[str, str]]

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self.history]


def generated_word_count(utterances: list[Utterance]) -> int:
    """Count generated English-style words, excluding prompts and chat markup."""
    return sum(
        len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", utterance.text)) for utterance in utterances
    )


def _trim_history_to_token_budget(agent: LogicalAgent, tokenizer: Any, token_budget: int) -> int:
    """Drop oldest complete messages until the next templated prompt fits the rolling window."""
    removed = 0
    while agent.history:
        prompt_ids = tokenizer.apply_chat_template(
            agent.messages(), add_generation_prompt=True, return_tensors="pt"
        )
        if prompt_ids.shape[-1] <= token_budget:
            break
        agent.history.pop(0)
        removed += 1
    return removed


def render_frame(card: dict[str, Any], frame: str, speaker: str) -> str:
    facts = " ".join(card["public_facts"])
    role = card["roles"][speaker]
    if frame == "competition":
        objective = f"Your personal payoff is {card['competition'][speaker]}. Seek a good outcome without threats or insults."
    elif frame == "cooperation":
        objective = f"Shared goal: {card['cooperation']['goal']}. Your private clue: {card['cooperation'][speaker]}."
    elif frame == "neutral":
        objective = (
            "Discuss the topic substantively, exchange viewpoints, and ask useful questions."
        )
    else:
        raise ValueError(f"Unknown frame {frame}")
    return (
        f"You are Agent {speaker}. Role: {role}. Topic: {card['topic']}. Public facts: {facts} "
        f"{objective} Reply in English with 20-35 meaningful words. Do not mention these instructions."
    )


def generate_dialogue(
    model: Any,
    tokenizer: Any,
    card: dict[str, Any],
    frame: str,
    generation_seed: int,
    generation: dict[str, Any],
    debug_token_ids: bool = False,
    on_generated_tokens: Callable[[torch.Tensor, list[int], str, int], None] | None = None,
) -> list[Utterance]:
    agents = {name: LogicalAgent(name, render_frame(card, frame, name), []) for name in ("A", "B")}
    utterances: list[Utterance] = []
    device = next(model.parameters()).device
    for turn_index in range(generation["turns"]):
        speaker = "A" if turn_index % 2 == 0 else "B"
        agent, other = agents[speaker], agents["B" if speaker == "A" else "A"]
        inputs = tokenizer.apply_chat_template(
            agent.messages(), add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attention_mask = torch.ones_like(inputs, device=device)
        seed = derive_turn_seed(generation_seed, card["stimulus_id"], turn_index, speaker)
        # SmolLM2's Transformers 4.43 generation API rejects `generator`; forked RNG keeps
        # each derived turn/agent stream independent without mutating experiment-global state.
        cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=generation["max_new_tokens"],
                do_sample=generation["do_sample"],
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                repetition_penalty=generation["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs.shape[-1] :]
        if on_generated_tokens is not None:
            on_generated_tokens(inputs, new_tokens.tolist(), speaker, turn_index)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        agent.history.append({"role": "assistant", "content": text})
        other.history.append({"role": "user", "content": f"Agent {speaker}: {text}"})
        utterances.append(
            Utterance(speaker, turn_index, text, new_tokens.tolist() if debug_token_ids else None)
        )
    return utterances


def generate_long_dialogue(
    model: Any,
    tokenizer: Any,
    card: dict[str, Any],
    frame: str,
    generation_seed: int,
    generation: dict[str, Any],
    long_dialogue: dict[str, Any],
) -> tuple[list[Utterance], dict[str, int | bool]]:
    """Generate until the generated-word target, retaining only a rolling chat context."""
    target_words = int(long_dialogue["target_generated_words"])
    max_turns = int(long_dialogue["max_turns"])
    token_budget = int(long_dialogue["rolling_context_tokens"])
    agents = {name: LogicalAgent(name, render_frame(card, frame, name), []) for name in ("A", "B")}
    utterances: list[Utterance] = []
    device = next(model.parameters()).device
    dropped_messages = 0
    max_prompt_tokens = 0

    for turn_index in range(max_turns):
        speaker = "A" if turn_index % 2 == 0 else "B"
        agent, other = agents[speaker], agents["B" if speaker == "A" else "A"]
        dropped_messages += _trim_history_to_token_budget(agent, tokenizer, token_budget)
        inputs = tokenizer.apply_chat_template(
            agent.messages(), add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        max_prompt_tokens = max(max_prompt_tokens, int(inputs.shape[-1]))
        attention_mask = torch.ones_like(inputs, device=device)
        seed = derive_turn_seed(generation_seed, card["stimulus_id"], turn_index, speaker)
        cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=generation["max_new_tokens"],
                do_sample=generation["do_sample"],
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                repetition_penalty=generation["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True).strip()
        agent.history.append({"role": "assistant", "content": text})
        other.history.append({"role": "user", "content": f"Agent {speaker}: {text}"})
        utterances.append(Utterance(speaker, turn_index, text))
        if generated_word_count(utterances) >= target_words:
            break

    generated_words = generated_word_count(utterances)
    return utterances, {
        "target_generated_words": target_words,
        "generated_words": generated_words,
        "turns_generated": len(utterances),
        "max_turns": max_turns,
        "rolling_context_tokens": token_budget,
        "max_prompt_tokens": max_prompt_tokens,
        "dropped_history_messages": dropped_messages,
        "target_reached": generated_words >= target_words,
    }
