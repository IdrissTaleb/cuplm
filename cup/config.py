"""Runtime configuration for Cup.

Defaults are tuned for the headline target: an old laptop with ~4GB RAM,
CPU-only. Everything is overridable via env vars or the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


@dataclass
class Config:
    # Path to a .gguf model. Empty => caller must use MockEngine or pass --mock.
    model_path: str = os.environ.get("CUP_MODEL", "")

    # Context window. 4096 is a safe floor for tiny models and keeps RAM low.
    n_ctx: int = _env_int("CUP_CTX", 4096)

    # CPU threads for llama.cpp. None => let llama.cpp pick (usually #cores).
    n_threads: "int | None" = None

    # Generation limits per step. A single Thought+Action rarely needs >256
    # tokens; capping it bounds wasted generation (the main per-step latency
    # cost after model size).
    max_tokens: int = _env_int("CUP_MAX_TOKENS", 256)
    temperature: float = 0.2  # agentic tasks want determinism; lower = less variance

    # Agent loop safety: hard cap on think->act cycles before we bail.
    max_steps: int = _env_int("CUP_MAX_STEPS", 8)

    # Shell skill: allow the agent to run commands. A deny-list still applies.
    allow_shell: bool = os.environ.get("CUP_ALLOW_SHELL", "1") != "0"
    shell_timeout: int = _env_int("CUP_SHELL_TIMEOUT", 30)

    # Root directory file/shell skills are confined to. Defaults to CWD.
    workdir: str = os.environ.get("CUP_WORKDIR", os.getcwd())

    # Print the agent's reasoning/actions as it works.
    verbose: bool = os.environ.get("CUP_VERBOSE", "1") != "0"

    # Disable a reasoning model's <think> phase (e.g. Qwen3) for speed.
    no_think: bool = os.environ.get("CUP_NO_THINK", "0") == "1"

    # Constrain decoding to valid ReAct via a GBNF grammar (llama.cpp server
    # only). Guarantees valid structure + real tool names, but on a weak stock
    # model it's score-neutral and slower, so it's opt-in. Expected to pay off
    # on the fine-tuned cuplm. Enable with `--grammar` or CUP_GRAMMAR=1.
    grammar: bool = os.environ.get("CUP_GRAMMAR", "0") == "1"

    # Use chat-template format (<|im_start|>/<|im_end|>) instead of raw text.
    # Required for instruct-tuned models (Qwen2.5-Coder-Instruct, Qwen3, etc.)
    # to correctly separate system/user/assistant roles. Enable with
    # --chat-format or CUP_CHAT_FORMAT=1.
    chat_format: bool = os.environ.get("CUP_CHAT_FORMAT", "0") == "1"
