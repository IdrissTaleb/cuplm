"""End-to-end proof of the agent loop using MockEngine (no model needed).

These tests pin down the behavior that makes Cup work on tiny models: tolerant
parsing, tool dispatch, observation injection, and clean termination.
"""

from __future__ import annotations

import os

from cup.agent import Agent
from cup.config import Config
from cup.engine import MockEngine
from cup.skills import build_default_registry
from cup.tools import parse


def _agent(responses, workdir):
    config = Config(workdir=str(workdir), verbose=False, allow_shell=True)
    registry = build_default_registry(config)
    return Agent(engine=MockEngine(responses), tools=registry, config=config)


def test_parse_action_json():
    r = parse('Thought: ok\nAction: read_file\nAction Input: {"path": "a.txt"}')
    assert r.kind == "action"
    assert r.tool_name == "read_file"
    assert r.tool_args == {"path": "a.txt"}


def test_parse_action_plain_string():
    r = parse("Action: list_dir\nAction Input: .")
    assert r.kind == "action"
    assert r.tool_args == "."


def test_parse_final():
    r = parse("Thought: done\nFinal Answer: 42 lines")
    assert r.kind == "final"
    assert r.final == "42 lines"


def test_parse_handles_code_fences():
    r = parse('Action: write_file\nAction Input: ```json\n{"path":"x","content":"y"}\n```')
    assert r.tool_args == {"path": "x", "content": "y"}


def test_final_answer_stops_at_trailing_ramble():
    # Real failure mode from Llama-3.2-3B: it keeps generating after the answer.
    text = (
        "Thought: done\n"
        "Final Answer: notes.txt has 2 lines.\n\n"
        "Note: paths are relative to the working directory.\n"
        "Thought: I should read the file first.\n"
        "Action: read_file\nAction Input: {\"path\": \"notes.txt\"}"
    )
    r = parse(text)
    assert r.kind == "final"
    assert r.final == "notes.txt has 2 lines."


def test_parse_collapses_duplicate_final_answer():
    r = parse("Final Answer: 3 lines.\nFinal Answer: 3 lines.")
    assert r.kind == "final"
    assert r.final == "3 lines."


def test_parse_stops_final_at_trailing_answer_label():
    r = parse("Final Answer: notes.txt has 3 lines.\n\nAnswer: 91 bytes.")
    assert r.final == "notes.txt has 3 lines."


def test_direct_answer_no_tool():
    # The model should be able to answer without any Action.
    config = Config(workdir=".", verbose=False)
    registry = build_default_registry(config)
    agent = Agent(engine=MockEngine(["Final Answer: My name is Cup."]), tools=registry, config=config)
    result = agent.run("What is your name?")
    assert result.stopped_reason == "final"
    assert result.answer == "My name is Cup."
    assert result.steps and result.steps[0].tool_name is None


def test_think_blocks_are_stripped():
    r = parse("<think>let me reason about this</think>\nAction: list_dir\nAction Input: .")
    assert r.kind == "action"
    assert r.tool_name == "list_dir"


def test_file_stats_counts_correctly(tmp_path):
    (tmp_path / "notes.txt").write_text("a b c\nd e\nf\n", encoding="utf-8")
    agent = _agent(
        [
            'Action: file_stats\nAction Input: {"path": "notes.txt"}',
            "Final Answer: 3 lines",
        ],
        tmp_path,
    )
    result = agent.run("how many lines and words?")
    obs = result.steps[0].observation or ""
    assert "lines=3" in obs and "words=6" in obs


def test_full_loop_reads_real_file(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    agent = _agent(
        [
            'Thought: read it.\nAction: read_file\nAction Input: {"path": "notes.txt"}',
            "Thought: two lines.\nFinal Answer: notes.txt has 2 lines.",
        ],
        tmp_path,
    )
    result = agent.run("How many lines in notes.txt?")
    assert result.stopped_reason == "final"
    assert "2 lines" in result.answer
    assert result.steps[0].tool_name == "read_file"
    assert "hello" in (result.steps[0].observation or "")


def test_full_loop_writes_file(tmp_path):
    agent = _agent(
        [
            'Action: write_file\nAction Input: {"path": "out.txt", "content": "hi there"}',
            "Final Answer: done",
        ],
        tmp_path,
    )
    result = agent.run("write hi there to out.txt")
    assert result.stopped_reason == "final"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi there"


def test_unknown_tool_is_recoverable(tmp_path):
    agent = _agent(
        [
            "Action: teleport\nAction Input: {}",
            "Final Answer: ok, used a real tool instead",
        ],
        tmp_path,
    )
    result = agent.run("do something")
    assert result.stopped_reason == "final"
    assert "unknown tool" in (result.steps[0].observation or "")


def test_path_escape_is_blocked(tmp_path):
    agent = _agent(
        [
            'Action: read_file\nAction Input: {"path": "../../secret"}',
            "Final Answer: could not read it",
        ],
        tmp_path,
    )
    result = agent.run("read the secret")
    assert "escapes" in (result.steps[0].observation or "")


def test_max_steps_guard(tmp_path):
    # Never emits a Final Answer -> must stop at max_steps, not hang.
    looping = ['Action: list_dir\nAction Input: {"path": "."}'] * 20
    config = Config(workdir=str(tmp_path), verbose=False, max_steps=3)
    registry = build_default_registry(config)
    agent = Agent(engine=MockEngine(looping), tools=registry, config=config)
    result = agent.run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3
