"""Contextual token cloud extraction for the dialogue-level PHD estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def dialogue_analysis_text(utterances: list[dict[str, Any]] | list[Any]) -> str:
    """Only generated text is admitted; no prompts, roles, labels, or stimuli."""
    return "\n".join(item["text"] if isinstance(item, dict) else item.text for item in utterances)


def load_analysis_encoder(analysis: dict[str, Any], device: str | None = None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        analysis["encoder_id"], revision=analysis["encoder_revision"]
    )
    model = AutoModel.from_pretrained(analysis["encoder_id"], revision=analysis["encoder_revision"])
    model.eval().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def contextual_token_embeddings(model: Any, tokenizer: Any, text: str) -> np.ndarray:
    """Return unnormalised, uncentred content-token embeddings in text order."""
    return contextual_token_cloud(model, tokenizer, text)[0]


@torch.inference_mode()
def contextual_token_cloud(
    model: Any, tokenizer: Any, text: str, maximum: int = 510
) -> tuple[np.ndarray, int, bool]:
    """Count all content tokens, then apply the preregistered text-order 510-token rule."""
    device = next(model.parameters()).device
    content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_full = len(content_ids)
    truncated = n_full > maximum
    input_ids = tokenizer.build_inputs_with_special_tokens(content_ids[:maximum])
    special = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    encoded = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = model(**encoded).last_hidden_state[0]
    content = hidden[~torch.tensor(special, device=device, dtype=torch.bool)]
    return content.detach().cpu().numpy().astype(np.float32, copy=False), n_full, truncated
