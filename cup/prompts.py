"""Prompt construction tuned for very small models.

Design notes (these matter a lot at 0.5B):
  * Keep the system prompt SHORT. Tiny models lose the plot in long preambles.
  * Show ONE concrete example. Few-shot beats instructions for small models.
  * Make the format mechanical and repetitive so the model can pattern-match.
"""

from __future__ import annotations

from cup.tools import ToolRegistry

# Kept deliberately short. Every token here is re-processed on every step, and
# tiny models focus better on less text. One worked example beats a wall of rules.
_SYSTEM_TEMPLATE = """You are Cup, a helpful assistant.

Answer the user DIRECTLY when you already know the answer or it is general
knowledge (math, definitions, how-to, chit-chat). Only use a tool when the task
truly needs it: reading or writing files, listing directories, or running a
command. Never use a tool to answer a general-knowledge question.

To answer directly, reply with one line:
Final Answer: <answer>

To use a tool:
Thought: <brief reasoning>
Action: <tool name>
Action Input: <input>
You then see "Observation: <result>", then continue. When done:
Final Answer: <answer>

Rules: use file facts, never guess counts. One action per step. Give the Final
Answer exactly once, then stop.

Tools:
{tool_list}

Example (general knowledge — NO tool):
Question: How do I define a function in Python?
Final Answer: Use the def keyword, e.g. `def add(a, b): return a + b`.

Example (chit-chat — NO tool):
Question: What is your name?
Final Answer: My name is Cup.

Example (needs a tool):
Question: How many lines are in notes.txt?
Thought: Use file_stats for the count.
Action: file_stats
Action Input: {{"path": "notes.txt"}}
Observation: lines=2 words=2 chars=12
Final Answer: notes.txt has 2 lines."""


def build_system_prompt(tools: ToolRegistry, no_think: bool = False) -> str:
    lines = []
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description} (input: {tool.args_hint})")
    tool_list = "\n".join(lines) if lines else "- (no tools available)"
    prompt = _SYSTEM_TEMPLATE.format(tool_list=tool_list)
    # Reasoning models (Qwen3) disable their <think> phase when they see this
    # directive — saves a lot of tokens/latency on weak CPUs.
    if no_think:
        prompt = "/no_think\n" + prompt
    return prompt


def build_prompt(system: str, question: str, scratchpad: str) -> str:
    """Assemble the full prompt the engine continues from."""
    return f"{system}\n\nQuestion: {question}\n{scratchpad}"
