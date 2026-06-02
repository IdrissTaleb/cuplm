"""Lock in the fine-tuning label-masking logic (no model/network needed).

Getting the mask wrong is the most common fine-tuning bug: if observations or
the prompt aren't masked, the model learns to hallucinate tool results. These
tests use a stub tokenizer so they run anywhere, instantly.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from training.train_qlora import tokenize_record  # noqa: E402


class StubTokenizer:
    """One integer id per whitespace token; deterministic and inspectable."""

    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}


def _record():
    return {
        "segments": [
            ["SYSTEM PROMPT here Question: q", 0],   # 5 tokens, masked
            ["Thought: act Action: read_file", 1],    # 4 tokens, trained
            ["\nObservation: lines=2\n", 0],          # 2 tokens, masked
            ["Final Answer: two", 1],                 # 3 tokens, trained
        ]
    }


def test_mask_trains_only_model_output():
    tok = StubTokenizer()
    ex = tokenize_record(tok, _record(), max_len=1000)
    # 5+4+2+3 content tokens + 1 EOS = 15
    assert len(ex["input_ids"]) == 15
    assert len(ex["labels"]) == 15
    # Trained = 4 (thought/action) + 3 (final) + 1 (eos) = 8
    trained = [l for l in ex["labels"] if l != -100]
    assert len(trained) == 8
    # The masked count = 5 (prompt) + 2 (observation) = 7
    assert sum(1 for l in ex["labels"] if l == -100) == 7


def test_eos_is_trained():
    tok = StubTokenizer()
    ex = tokenize_record(tok, _record(), max_len=1000)
    assert ex["input_ids"][-1] == tok.eos_token_id
    assert ex["labels"][-1] == tok.eos_token_id


def test_truncation_respects_max_len():
    tok = StubTokenizer()
    ex = tokenize_record(tok, _record(), max_len=6)
    assert len(ex["input_ids"]) == 6
    assert len(ex["labels"]) == 6
