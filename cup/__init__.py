"""Cup — an agentic framework for tiny language models on weak/legacy CPUs.

Cup runs `cuplm` (or any GGUF model) through llama.cpp and drives a
reliable, tolerant agent loop designed for the failure modes of very small
models. The goal: capable agentic AI, fully offline, on hardware everyone
already owns.
"""

from cup.agent import Agent
from cup.config import Config
from cup.engine import (
    Engine,
    LlamaCppEngine,
    LlamaServerEngine,
    MockEngine,
    TransformersEngine,
)
from cup.tools import Tool, ToolRegistry

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "Config",
    "Engine",
    "LlamaCppEngine",
    "LlamaServerEngine",
    "TransformersEngine",
    "MockEngine",
    "Tool",
    "ToolRegistry",
    "__version__",
]
