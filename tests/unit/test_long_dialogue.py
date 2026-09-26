import torch
from agent_language_geometry.dialogue import (
    LogicalAgent,
    _trim_history_to_token_budget,
    generated_word_count,
)
from agent_language_geometry.schemas import Utterance


class _Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        del add_generation_prompt, return_tensors
        return torch.ones(
            (1, sum(len(message["content"].split()) for message in messages)), dtype=torch.long
        )


def test_generated_word_count_excludes_punctuation_and_counts_contractions() -> None:
    utterances = [Utterance("A", 0, "Hello, we're testing 42 tokens."), Utterance("B", 1, "OK!")]
    assert generated_word_count(utterances) == 5


def test_rolling_context_drops_oldest_complete_messages() -> None:
    agent = LogicalAgent(
        "A",
        "system words",
        [
            {"role": "user", "content": "one two three"},
            {"role": "assistant", "content": "four five six"},
        ],
    )
    assert _trim_history_to_token_budget(agent, _Tokenizer(), 5) == 1
    assert agent.history == [{"role": "assistant", "content": "four five six"}]
