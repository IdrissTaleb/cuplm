"""Generate self-verifying SFT data for fine-tuning `cuplm`.

The trick: we don't trust a teacher model to produce correct tool-use traces.
Instead we *construct* tasks with known answers, run them through Cup's REAL
tools to get ground-truth observations, and assemble perfect ReAct trajectories
in the EXACT format Cup uses at inference (train/inference parity).

We also emit "direct answer" examples (identity, math, how-to) so the model
learns NOT to reach for a tool on general-knowledge questions — directly fixing
the two weaknesses we saw on stock Qwen3-0.6B.

Output: training/data/cuplm_sft.jsonl
Each record:
  {
    "question": str,
    "answer": str,
    "text": str,                       # full formatted sequence
    "segments": [[text, trainable], …] # 1 = train on it (model output), 0 = mask
  }
Run from the repo root:  python -m training.generate_data
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

from cup.config import Config
from cup.prompts import build_system_prompt
from cup.skills import build_default_registry

random.seed(7)

OUT = Path(__file__).parent / "data" / "cuplm_sft.jsonl"
NO_THINK = True  # match the recommended `cup ... --no-think` invocation for Qwen3

# One system prompt, identical to inference, reused for every example.
_TMP_CFG = Config(workdir=tempfile.gettempdir(), no_think=NO_THINK)
SYSTEM = build_system_prompt(build_default_registry(_TMP_CFG), no_think=NO_THINK)


def _tool(workdir: str, name: str, args):
    """Run a real Cup tool and return its exact observation string."""
    reg = build_default_registry(Config(workdir=workdir, no_think=NO_THINK))
    return reg.get(name).run(args)


def _assemble(question: str, moves: list[tuple[str, str]]) -> dict:
    """moves: ordered list of ("model", text) and ("obs", text).
    Builds the full text + per-segment training mask, matching cup.agent /
    cup.prompts exactly (scratchpad appends '\\nObservation: {obs}\\n')."""
    segments: list[list] = [[f"{SYSTEM}\n\nQuestion: {question}\n", 0]]
    answer = ""
    for kind, text in moves:
        if kind == "model":
            segments.append([text, 1])
            if text.lower().startswith("final answer"):
                answer = text.split(":", 1)[1].strip()
        else:  # observation: model must NOT learn to generate these
            segments.append([f"\nObservation: {text}\n", 0])
    full = "".join(s[0] for s in segments)
    return {"question": question, "answer": answer, "text": full, "segments": segments}


# --- Tool-use tasks (observations come from real tools) --------------------

_FILE_BODIES = [
    "alpha\nbeta\ngamma\n",
    "one line only\n",
    "to do:\n- buy milk\n- call mom\n- ship cuplm\n",
    "name,age\nAda,36\nAlan,41\nGrace,52\n",
    "The quick brown fox\njumps over\nthe lazy dog\n",
    "",
]
_FNAMES = ["notes.txt", "todo.txt", "data.csv", "log.txt", "readme.md"]


def gen_count_tasks(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            fname = random.choice(_FNAMES)
            body = random.choice(_FILE_BODIES)
            Path(wd, fname).write_text(body, encoding="utf-8")
            obs = _tool(wd, "file_stats", {"path": fname})  # lines=.. words=.. chars=.. bytes=..
            stats = dict(p.split("=") for p in obs.split())
            metric, word = random.choice([("lines", "lines"), ("words", "words"), ("chars", "characters")])
            q = random.choice([
                f"How many {word} are in {fname}?",
                f"Count the {word} in {fname}.",
                f"{fname} - how many {word}?",
            ])
            thought = f"Thought: Use file_stats to get the count.\nAction: file_stats\nAction Input: {{\"path\": \"{fname}\"}}"
            final = f"Final Answer: {fname} has {stats[metric]} {word}."
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


def gen_list_tasks(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            k = random.randint(1, 5)
            names = random.sample(_FNAMES, k)
            for nm in names:
                Path(wd, nm).write_text("x", encoding="utf-8")
            obs = _tool(wd, "list_dir", {"path": "."})
            count = len([ln for ln in obs.splitlines() if ln.strip()])
            q = random.choice([
                "How many files are in this folder?",
                "List the files here and tell me how many there are.",
                "What's in the current directory?",
            ])
            thought = "Thought: List the directory to see the files.\nAction: list_dir\nAction Input: {\"path\": \".\"}"
            final = f"Final Answer: There are {count} item(s) here: " + ", ".join(names) + "."
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


def gen_read_tasks(n: int) -> list[dict]:
    out = []
    facts = [
        ("config.ini", "port=8080\nhost=localhost\n", "What port is in config.ini?", "The port is 8080."),
        ("version.txt", "cuplm v0.1\n", "What version is in version.txt?", "It is cuplm v0.1."),
        ("owner.txt", "owner: devou\n", "Who is the owner per owner.txt?", "The owner is devou."),
    ]
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            fname, body, q, ans = random.choice(facts)
            Path(wd, fname).write_text(body, encoding="utf-8")
            obs = _tool(wd, "read_file", {"path": fname})
            thought = f"Thought: Read the file to find the answer.\nAction: read_file\nAction Input: {{\"path\": \"{fname}\"}}"
            final = f"Final Answer: {ans}"
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


def gen_write_tasks(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            fname = random.choice(["hello.txt", "out.txt", "greeting.txt"])
            content = random.choice(["hello world", "cup is running", "done!"])
            q = f'Create a file called {fname} that says "{content}".'
            obs = _tool(wd, "write_file", {"path": fname, "content": content})
            thought = (
                f"Thought: Write the content to the file.\nAction: write_file\n"
                f"Action Input: {{\"path\": \"{fname}\", \"content\": \"{content}\"}}"
            )
            final = f"Final Answer: Created {fname}."
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


# --- Direct-answer tasks (NO tool) -----------------------------------------

def gen_identity(n: int) -> list[dict]:
    qs = [
        "What's your name?", "Who are you?", "What are you called?",
        "Introduce yourself.", "What is your name?", "hey, what's your name?",
    ]
    ans = "My name is Cup. I'm a small AI agent that runs offline on your computer."
    return [_assemble(random.choice(qs), [("model", f"Final Answer: {ans}")]) for _ in range(n)]


def gen_capabilities(n: int) -> list[dict]:
    qs = ["What can you do?", "What are you for?", "How can you help me?"]
    ans = ("I can read and write files, list folders, run commands, and answer "
           "questions - all locally on your machine.")
    return [_assemble(random.choice(qs), [("model", f"Final Answer: {ans}")]) for _ in range(n)]


def gen_math(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        a, b = random.randint(0, 50), random.randint(0, 50)
        op, val = random.choice([("+", a + b), ("-", a - b), ("*", a * b)])
        q = f"What is {a} {op} {b}?"
        out.append(_assemble(q, [("model", f"Final Answer: {a} {op} {b} = {val}.")]))
    return out


def gen_howto(n: int) -> list[dict]:
    pairs = [
        ("How do I define a function in Python?",
         "Use the def keyword, e.g. `def add(a, b): return a + b`."),
        ("How do I write a for loop in Python?",
         "Use `for x in iterable:` then an indented body, e.g. `for i in range(3): print(i)`."),
        ("How do I open a file in Python?",
         "Use `with open('file.txt') as f: data = f.read()`."),
        ("How do I print in Python?", "Call `print(\"hello\")`."),
        ("How do I make a list in Python?", "Use square brackets, e.g. `nums = [1, 2, 3]`."),
    ]
    out = []
    for _ in range(n):
        q, a = random.choice(pairs)
        out.append(_assemble(q, [("model", f"Final Answer: {a}")]))
    return out


def gen_unknown(n: int) -> list[dict]:
    qs = ["What's the weather tomorrow?", "What did I have for lunch?", "What's my password?"]
    ans = "I can't know that - I run locally and don't have access to that information."
    return [_assemble(random.choice(qs), [("model", f"Final Answer: {ans}")]) for _ in range(n)]


def main() -> None:
    records: list[dict] = []
    records += gen_count_tasks(120)
    records += gen_list_tasks(80)
    records += gen_read_tasks(60)
    records += gen_write_tasks(60)
    records += gen_identity(40)
    records += gen_capabilities(20)
    records += gen_math(120)
    records += gen_howto(60)
    records += gen_unknown(20)
    random.shuffle(records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    tool_ct = sum(1 for r in records if any(s[0].startswith("\nObservation") for s in r["segments"]))
    print(f"Wrote {len(records)} examples to {OUT}")
    print(f"  tool-use trajectories: {tool_ct}")
    print(f"  direct-answer examples: {len(records) - tool_ct}")
    print("\n--- sample (tool-use) ---")
    sample = next(r for r in records if any(s[0].startswith('\nObservation') for s in r['segments']))
    print(sample["text"][-400:])


if __name__ == "__main__":
    main()
