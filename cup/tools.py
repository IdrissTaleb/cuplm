"""Tools, the tool registry, and a tolerant parser for the agent protocol.

Tiny models are *unreliable* at strict formats (especially JSON). The whole
design philosophy here is **be forgiving**: accept messy output, extract intent,
and only fail when there's truly nothing usable. This is the single biggest
lever for making a 0.5B model behave like an agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

# A tool receives either a parsed dict of args or a raw string, and returns a
# string observation.
ToolArgs = Union[dict, str]
ToolFunc = Callable[[ToolArgs], str]


@dataclass
class Tool:
    """A single capability the agent can invoke."""

    name: str
    description: str
    # Human/LLM-readable hint about expected arguments, shown in the prompt.
    args_hint: str
    func: ToolFunc

    def run(self, args: ToolArgs) -> str:
        try:
            return self.func(args)
        except Exception as exc:  # tools must never crash the loop
            return f"ERROR: {type(exc).__name__}: {exc}"


class ToolRegistry:
    """An ordered collection of tools, addressable by name."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# --- Parsing the model's output -------------------------------------------

# We support a classic, small-model-friendly ReAct format:
#
#   Thought: <reasoning>
#   Action: <tool_name>
#   Action Input: <json or plain text>
#
# ...or a terminal:
#
#   Thought: <reasoning>
#   Final Answer: <answer to the user>

# Capture the final answer but STOP if the (often over-eager) small model keeps
# generating a new section afterwards — a fresh Thought/Action/Note/Question line
# or a blank line ends the answer. Without this, trailing rambling leaks into the
# answer (observed with Llama-3.2-3B).
_FINAL_RE = re.compile(
    r"final\s*answer\s*:\s*(.*?)"
    # Stop at: a new section, a REPEATED "Final Answer"/"Answer:" line (small
    # models love to restate the answer), a blank line, or end of text.
    r"(?:\n\s*(?:thought|action|observation|note|question|final\s*answer|answer)\s*[:\-]"
    r"|\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_RE = re.compile(r"action\s*:\s*([^\n]+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(
    r"action\s*input\s*:\s*(.*?)(?:\n\s*(?:observation|thought)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParseResult:
    kind: str  # "final" | "action" | "none"
    final: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[ToolArgs] = None


def _coerce_args(raw: str) -> ToolArgs:
    """Turn the Action Input text into a dict if it looks like JSON, else keep
    it as a stripped string. Forgiving by design."""
    raw = raw.strip()
    if not raw:
        return {}
    # Strip code fences a small model may emit.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    if (raw.startswith("{") and raw.endswith("}")) or (
        raw.startswith("[") and raw.endswith("]")
    ):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass  # fall through to raw string
    return raw


# Reasoning models (e.g. Qwen3) wrap private reasoning in <think>...</think>.
# Strip it before parsing so it never interferes with action/answer extraction.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse(text: str) -> ParseResult:
    """Parse a model completion into the next agent step.

    Final answer wins if present (a small model sometimes emits both an action
    and a final answer; the answer is the safer terminal choice only if no
    action precedes it — so we check action first when both appear inline)."""
    text = _THINK_RE.sub("", text)
    # If the model produced an Action, prefer acting (unless a Final Answer
    # clearly comes first in the text).
    action_m = _ACTION_RE.search(text)
    final_m = _FINAL_RE.search(text)

    if action_m and (not final_m or action_m.start() < final_m.start()):
        tool_name = action_m.group(1).strip().strip("`\"' ")
        input_m = _ACTION_INPUT_RE.search(text)
        args: ToolArgs = _coerce_args(input_m.group(1)) if input_m else {}
        return ParseResult(kind="action", tool_name=tool_name, tool_args=args)

    if final_m:
        return ParseResult(kind="final", final=final_m.group(1).strip())

    return ParseResult(kind="none")
