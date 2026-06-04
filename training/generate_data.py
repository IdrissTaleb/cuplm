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


def gen_code_write(n: int) -> list[dict]:
    """Direct code generation answers — NO tool, model writes code from memory."""
    pairs = [
        ("Write a Python function that adds two numbers.",
         "def add(a, b):\n    return a + b"),
        ("Write a Python function that reverses a string.",
         "def reverse(s):\n    return s[::-1]"),
        ("Write a Python function that checks if a number is even.",
         "def is_even(n):\n    return n % 2 == 0"),
        ("Write a Python function that returns the factorial of n.",
         "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)"),
        ("Write a Python function that finds the maximum in a list.",
         "def find_max(lst):\n    return max(lst)"),
        ("Write a Python class for a simple stack.",
         "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)\n    def pop(self): return self.items.pop()\n    def is_empty(self): return not self.items"),
        ("How do I read a JSON file in Python?",
         "import json\nwith open('file.json') as f:\n    data = json.load(f)"),
        ("How do I sort a list of dicts by a key in Python?",
         "sorted(items, key=lambda x: x['key'])"),
        ("Write a Python function that checks if a string is a palindrome.",
         "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
        ("How do I make an HTTP GET request in Python?",
         "import urllib.request\nwith urllib.request.urlopen('http://example.com') as r:\n    data = r.read().decode()"),
        ("Write a Python function to count word frequency in a string.",
         "def word_freq(text):\n    words = text.lower().split()\n    freq = {}\n    for w in words: freq[w] = freq.get(w, 0) + 1\n    return freq"),
        ("Write a Python function that flattens a nested list.",
         "def flatten(lst):\n    out = []\n    for item in lst:\n        if isinstance(item, list): out.extend(flatten(item))\n        else: out.append(item)\n    return out"),
    ]
    out = []
    for _ in range(n):
        q, code = random.choice(pairs)
        formatted = f"```python\n{code}\n```"
        out.append(_assemble(q, [("model", f"Final Answer: {formatted}")]))
    return out


def gen_run_command(n: int) -> list[dict]:
    """Tasks that need run_command to check the environment."""
    pairs = [
        ("What Python version is installed?",
         "python --version", "Python"),
        ("List all .txt files in the current directory.",
         "dir *.txt" if os.name == "nt" else "ls *.txt", "txt"),
        ("Create a directory called output.",
         "mkdir output", ""),
    ]
    out = []
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            q, cmd, _ = random.choice(pairs)
            try:
                import subprocess
                result = subprocess.run(
                    cmd, shell=True, cwd=wd, capture_output=True, text=True, timeout=10
                )
                obs = (result.stdout + result.stderr).strip() or "(no output)"
            except Exception as e:
                obs = f"ERROR: {e}"
            thought = (f"Thought: Run a command to find out.\n"
                       f"Action: run_command\n"
                       f'Action Input: {{"command": "{cmd}"}}')
            final = f"Final Answer: {obs[:120]}"
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


def gen_write_code_to_file(n: int) -> list[dict]:
    """Write a Python script to a file using write_file — multi-step code tasks."""
    tasks = [
        ("hello.py", "print('Hello, World!')", "Write a Python hello world script and save it to hello.py."),
        ("add.py", "def add(a, b):\n    return a + b\n\nprint(add(3, 4))", "Save a Python script that adds 3 and 4 to add.py."),
        ("greet.py", "name = input('Name: ')\nprint(f'Hello, {name}!')", "Write a greeting script to greet.py that asks for a name."),
        ("counter.py", "for i in range(1, 6):\n    print(i)", "Write a Python script that prints 1 through 5 and save it as counter.py."),
    ]
    out = []
    for _ in range(n):
        with tempfile.TemporaryDirectory() as wd:
            fname, code, q = random.choice(tasks)
            obs = _tool(wd, "write_file", {"path": fname, "content": code})
            thought = (
                f"Thought: Write the Python code to the file.\n"
                f"Action: write_file\n"
                f'Action Input: {{"path": "{fname}", "content": {json.dumps(code)}}}'
            )
            final = f"Final Answer: Saved the script to {fname}."
            out.append(_assemble(q, [("model", thought), ("obs", obs), ("model", final)]))
    return out


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
    records += gen_code_write(120)       # new: direct code generation
    records += gen_write_code_to_file(60)  # new: write code to file
    records += gen_run_command(20)       # new: shell command tasks
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
