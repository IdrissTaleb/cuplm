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
_SYSTEM_TEMPLATE = """You are Cup, an offline AI assistant.

Answer from your own knowledge for general questions: math, definitions, coding
help, greetings, how-to. Only call a tool when the task truly requires it —
reading/writing files, listing directories, or running a command.

RULES:
- For general knowledge (math, definitions, coding help, greetings): answer directly with Final Answer.
- For anything about a specific file or directory on disk: ALWAYS call the right tool first. Never guess file contents, line counts, or directory listings.
- One tool call per step. Write Final Answer exactly once, then stop.

Tools:
{tool_list}

Examples:

Question: How do I define a function in Python?
Final Answer: Use the def keyword, e.g. `def add(a, b): return a + b`

Question: What is your name?
Final Answer: My name is Cup, an offline AI assistant.

Question: What is 12 times 7?
Final Answer: 84

Question: How many lines are in report.txt?
Thought: I must call file_stats to get the exact count, not guess.
Action: file_stats
Action Input: {{"path": "report.txt"}}
Observation: lines=47 words=312 chars=1840 bytes=1840
Final Answer: report.txt has 47 lines.

Question: List the files in this folder.
Thought: I will call list_dir to see the directory.
Action: list_dir
Action Input: {{"path": "."}}
Observation: file  a.txt
file  b.txt
file  c.txt
Final Answer: There are 3 files: a.txt, b.txt, c.txt.

Question: Write hello world to hello.txt
Thought: I will write the file using write_file.
Action: write_file
Action Input: {{"path": "hello.txt", "content": "hello world"}}
Observation: Wrote 11 chars to hello.txt.
Final Answer: Done, wrote hello world to hello.txt."""


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
    """Assemble the full prompt the engine continues from (raw text / base model)."""
    return f"{system}\n\nQuestion: {question}\n{scratchpad}"


def build_prompt_chat(system: str, question: str, scratchpad: str) -> str:
    """Chat-template format for instruct models (Qwen2.5, Qwen3, etc.).

    The system prompt (tools + examples) goes in <|im_start|>system so the model
    treats it as background knowledge, not as part of the conversation. The
    user question is a proper user turn. The scratchpad pre-fills the assistant
    turn so the model continues the ReAct trace from the right position.

    On the first step (empty scratchpad) we prime with 'Thought:' to force the
    model into reasoning mode instead of blurting out a direct answer.
    """
    assistant_prefix = scratchpad if scratchpad else "Thought:"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )
