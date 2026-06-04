"""Command-line interface: `cup`.

Examples:
  cup chat                                  # interactive — just type questions
  cup run "list the files here"             # one-shot, prints only the answer
  cup run "summarize notes.txt" --info      # also show the reasoning
  cup run "count lines in a.txt" --background  # answer + timing + tool trace
  cup --version

Set CUP_SERVER (e.g. http://localhost:8080) and CUP_WORKDIR once and you can
just run `cup chat` with no flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from cup import __version__
from cup.agent import Agent
from cup.config import Config
from cup.engine import LlamaCppEngine, LlamaServerEngine, MockEngine, TransformersEngine
from cup.skills import build_default_registry


def _demo_mock_engine() -> MockEngine:
    """A scripted run so `--mock` does something visible without a model."""
    return MockEngine(
        [
            'Thought: see what files are here.\nAction: list_dir\nAction Input: {"path": "."}',
            "Final Answer: I listed the current directory.",
        ]
    )


def build_agent(args) -> Agent:
    config = Config()
    if getattr(args, "model", None):
        config.model_path = args.model
    if getattr(args, "workdir", None):
        config.workdir = args.workdir
    if getattr(args, "no_shell", False):
        config.allow_shell = False
    if getattr(args, "no_think", False):
        config.no_think = True
    if getattr(args, "grammar", False):
        config.grammar = True
    if getattr(args, "chat_format", False):
        config.chat_format = True
    # Quiet by default; reasoning is shown only when asked (--info/--background,
    # or per-line in chat mode).
    config.verbose = bool(getattr(args, "info", False) or getattr(args, "background", False))

    registry = build_default_registry(config)

    if getattr(args, "mock", False):
        engine = _demo_mock_engine()
    elif getattr(args, "server", None):
        engine = LlamaServerEngine(base_url=args.server)
    elif getattr(args, "hf", None):
        engine = TransformersEngine(args.hf, adapter=getattr(args, "adapter", None))
    elif config.model_path:
        engine = LlamaCppEngine(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            verbose=False,
        )
    else:
        print(
            "No engine selected. Options:\n"
            "  --server http://localhost:8080   (recommended; or set CUP_SERVER)\n"
            "  --model path/to/model.gguf       (in-process, needs a compatible wheel)\n"
            "  --mock                           (no model; exercise the agent loop)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return Agent(engine=engine, tools=registry, config=config)


def _print_background(result, elapsed: float) -> None:
    print(f"\n[background] {len(result.steps)} step(s) · {elapsed:.1f}s · stopped={result.stopped_reason}")
    for i, step in enumerate(result.steps, 1):
        if step.tool_name:
            obs = (step.observation or "").replace("\n", " ")
            if len(obs) > 80:
                obs = obs[:80] + "..."
            print(f"  {i}. {step.tool_name}({step.tool_args}) -> {obs}")


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="Path to a .gguf model (or set CUP_MODEL)")
    p.add_argument(
        "--server",
        default=os.environ.get("CUP_SERVER"),
        help="URL of a running llama.cpp server (or set CUP_SERVER)",
    )
    p.add_argument("--workdir", help="Directory the agent is confined to (default: CWD)")
    p.add_argument("--hf", help="Run a HuggingFace model in-process (id or path), e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", help="LoRA adapter dir to load on top of --hf (a trained cuplm)")
    p.add_argument("--mock", action="store_true", help="Use the scripted mock engine")
    p.add_argument("--no-shell", action="store_true", help="Disable the run_command tool")
    p.add_argument(
        "--no-think", action="store_true",
        help="Disable a reasoning model's thinking phase (recommended for Qwen3)",
    )
    p.add_argument(
        "--grammar", action="store_true",
        help="Force valid ReAct output via GBNF grammar (llama.cpp server; best with cuplm)",
    )
    p.add_argument(
        "--chat-format", action="store_true", dest="chat_format",
        help="Use chat-template format for instruct models (Qwen2.5-Coder, Qwen3, etc.)",
    )


def cmd_run(args) -> int:
    show = args.info or args.background
    agent = build_agent(args)
    if show:
        print(f"> {args.task}\n" + "-" * 60)
    start = time.perf_counter()
    result = agent.run(args.task)
    elapsed = time.perf_counter() - start
    if show:
        print("-" * 60)
    print(result.answer.strip())  # the headline: just the answer
    if args.background:
        _print_background(result, elapsed)
    return 0


def cmd_chat(args) -> int:
    agent = build_agent(args)
    print("Cup chat — just type a question and press Enter.")
    print("Prefix a line with --info or --background to see the work. Type 'exit' to quit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", ":q"):
            break

        show_info = show_bg = False
        if line.startswith("--background"):
            show_bg, line = True, line[len("--background"):].strip()
        elif line.startswith("--info"):
            show_info, line = True, line[len("--info"):].strip()
        if not line:
            continue

        agent.config.verbose = show_info or show_bg
        start = time.perf_counter()
        result = agent.run(line)
        elapsed = time.perf_counter() - start
        print(f"cup> {result.answer.strip()}")
        if show_bg:
            _print_background(result, elapsed)
    return 0


def main(argv=None) -> int:
    # Windows consoles default to cp1252 and choke on UTF-8 model output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(prog="cup", description="Agentic AI for weak/legacy CPUs.")
    parser.add_argument("--version", action="version", version=f"cup {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the agent on a single task")
    run.add_argument("task", help="What you want the agent to do")
    _add_engine_args(run)
    run.add_argument("--info", action="store_true", help="Show the agent's reasoning trace")
    run.add_argument("--background", action="store_true", help="Show reasoning plus timing and a tool trace")

    chat = sub.add_parser("chat", help="Interactive chat; just type questions")
    _add_engine_args(chat)

    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    if args.command == "chat":
        return cmd_chat(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
