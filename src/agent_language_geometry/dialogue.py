"""Deterministic two-logical-agent dialogue protocol with separate histories/RNGs."""

from __future__ import annotations

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
