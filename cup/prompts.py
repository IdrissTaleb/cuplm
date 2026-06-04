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
_SYSTEM_TEMPLATE = """You are Cup, a helpful assistant that runs fully offline.

Answer directly from your own knowledge for general questions (math, definitions,
how-to, greetings, coding help). Only call a tool when the task genuinely requires
it: reading/writing files, listing directories, or running a shell command.
Never call a tool to answer a general-knowledge or coding question.

Format for a direct answer (no tool needed):
Final Answer: the answer text

Format for a tool call:
Thought: one-sentence reason you need this tool
Action: tool_name
Action Input: {{"key": "value"}}
After the tool runs you see:
Observation: the result
Then continue reasoning. End with:
Final Answer: the answer text

Rules: one tool call per step. Give Final Answer exactly once, then stop.
Never output tags like <answer> or <input> — write the real text directly.

Tools:
{tool_list}

Example (coding question — NO tool):
Question: How do I define a function in Python?
Final Answer: Use the def keyword: `def add(a, b): return a + b`

Example (greeting — NO tool):
Question: What is your name?
Final Answer: My name is Cup, an offline AI assistant.

Example (math — NO tool):
Question: What is 12 times 7?
Final Answer: 84

Example (needs a tool):
Question: How many lines are in notes.txt?
Thought: I need the exact count from the file, not a guess.
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
