"""Benchmark tasks for the Cup agent framework.

Each Task bundles four things:
  * a prompt (the instruction handed to the agent),
  * a setup callback that seeds the temp workdir with any needed files,
  * a check callback that, given the workdir and the agent Result, decides
    whether the run succeeded, and
  * a list of scripted MockEngine responses that make the task pass with no
    real model (used by `--mock`).

The checks are deliberately tolerant: tiny models phrase answers loosely, so we
lean on case-insensitive substring matches and integer extraction rather than
exact equality.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, List

# Result is only imported for typing; avoid a hard import cycle at runtime.
try:  # pragma: no cover - typing convenience only
    from cup.agent import Result
except Exception:  # pragma: no cover
    Result = object  # type: ignore


@dataclass
class Task:
    name: str
    prompt: str
    setup: Callable[[str], None]
    check: Callable[[str, "Result"], bool]
    # Scripted completions for MockEngine, replayed in order across agent steps.
    # These let `--mock` exercise the whole harness without a model.
    mock_responses: List[str] = field(default_factory=list)


# --- helpers ---------------------------------------------------------------


def extract_ints(text: str) -> List[int]:
    """Pull every integer out of a string (handles commas like 1,234)."""
    if not text:
        return []
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    return [int(m) for m in re.findall(r"-?\d+", cleaned)]


def answer_has_int(answer: str, target: int) -> bool:
    """True if `target` appears as one of the integers in the answer."""
    return target in extract_ints(answer)


def answer_contains(answer: str, needle: str) -> bool:
    """Case-insensitive substring match, tolerant of None."""
    return needle.lower() in (answer or "").lower()


def read(workdir: str, rel: str) -> str:
    path = os.path.join(workdir, rel)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write(workdir: str, rel: str, content: str) -> None:
    path = os.path.join(workdir, rel)
    os.makedirs(os.path.dirname(path) or workdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# --- task definitions ------------------------------------------------------


def _setup_count_lines(workdir: str) -> None:
    # 5 lines, no trailing newline ambiguity beyond the final \n.
    write(workdir, "notes.txt", "alpha\nbeta\ngamma\ndelta\nepsilon\n")


def _check_count_lines(workdir: str, result) -> bool:
    return answer_has_int(result.answer, 5)


def _setup_largest_file(workdir: str) -> None:
    write(workdir, "small.txt", "x" * 10)
    write(workdir, "medium.txt", "y" * 100)
    write(workdir, "big.txt", "z" * 1000)


def _check_largest_file(workdir: str, result) -> bool:
    return answer_contains(result.answer, "big.txt")


def _setup_report_fact(workdir: str) -> None:
    write(
        workdir,
        "config.ini",
        "[server]\nhost = example.com\nport = 8042\ntimeout = 30\n",
    )


def _check_report_fact(workdir: str, result) -> bool:
    return answer_has_int(result.answer, 8042)


def _setup_write_file(workdir: str) -> None:
    # Nothing to seed; the agent must create the file.
    return None


def _check_write_file(workdir: str, result) -> bool:
    return "hello cup" in read(workdir, "greeting.txt").lower()


def _setup_list_count(workdir: str) -> None:
    write(workdir, "a.txt", "1")
    write(workdir, "b.txt", "2")
    write(workdir, "c.txt", "3")
    write(workdir, "d.txt", "4")


def _check_list_count(workdir: str, result) -> bool:
    return answer_has_int(result.answer, 4)


def _setup_roundtrip(workdir: str) -> None:
    return None


def _check_roundtrip(workdir: str, result) -> bool:
    # File must have been created with the secret, and the answer must report it.
    file_ok = "swordfish" in read(workdir, "secret.txt").lower()
    answer_ok = answer_contains(result.answer, "swordfish")
    return file_ok and answer_ok


TASKS: List[Task] = [
    Task(
        name="count_lines",
        prompt="How many lines are in notes.txt? Reply with the number.",
        setup=_setup_count_lines,
        check=_check_count_lines,
        mock_responses=[
            'Thought: read the file.\nAction: read_file\nAction Input: {"path": "notes.txt"}',
            "Thought: it has five lines.\nFinal Answer: notes.txt has 5 lines.",
        ],
    ),
    Task(
        name="largest_file",
        prompt="Which file in the current directory is the largest? Name it.",
        setup=_setup_largest_file,
        check=_check_largest_file,
        mock_responses=[
            'Thought: inspect sizes.\nAction: run_command\nAction Input: {"command": "ls"}',
            "Thought: big.txt is clearly the biggest.\nFinal Answer: The largest file is big.txt.",
        ],
    ),
    Task(
        name="report_fact",
        prompt="Read config.ini and tell me what port the server uses.",
        setup=_setup_report_fact,
        check=_check_report_fact,
        mock_responses=[
            'Thought: read it.\nAction: read_file\nAction Input: {"path": "config.ini"}',
            "Thought: the port line says 8042.\nFinal Answer: The server uses port 8042.",
        ],
    ),
    Task(
        name="write_file",
        prompt="Create a file named greeting.txt containing exactly: hello cup",
        setup=_setup_write_file,
        check=_check_write_file,
        mock_responses=[
            'Thought: write the file.\nAction: write_file\nAction Input: {"path": "greeting.txt", "content": "hello cup"}',
            "Thought: done.\nFinal Answer: I created greeting.txt with the text 'hello cup'.",
        ],
    ),
    Task(
        name="list_count",
        prompt="How many entries are in the current directory? Reply with the number.",
        setup=_setup_list_count,
        check=_check_list_count,
        mock_responses=[
            'Thought: list it.\nAction: list_dir\nAction Input: {"path": "."}',
            "Thought: there are four files.\nFinal Answer: There are 4 entries.",
        ],
    ),
    Task(
        name="create_then_read",
        prompt=(
            "Write the word swordfish into a file named secret.txt, then read "
            "secret.txt back and tell me what it contains."
        ),
        setup=_setup_roundtrip,
        check=_check_roundtrip,
        mock_responses=[
            'Thought: write it.\nAction: write_file\nAction Input: {"path": "secret.txt", "content": "swordfish"}',
            'Thought: now read it back.\nAction: read_file\nAction Input: {"path": "secret.txt"}',
            "Thought: it says swordfish.\nFinal Answer: secret.txt contains the word swordfish.",
        ],
    ),
]
