"""Command-line interface: `cup`.

Examples:
  cup run "list the files here and tell me which is largest" --model path/to/cuplm.gguf
  cup run "summarize notes.txt" --mock          # exercise the loop, no model
  cup --version
"""

from __future__ import annotations

import argparse
import sys

from cup import __version__
from cup.agent import Agent
from cup.config import Config
from cup.engine import LlamaCppEngine, LlamaServerEngine, MockEngine
from cup.skills import build_default_registry


def _demo_mock_engine() -> MockEngine:
    """A tiny scripted run so `--mock` does something visible end-to-end."""
    return MockEngine(
        [
            "Thought: I should see what files are here.\n"
            "Action: list_dir\nAction Input: {\"path\": \".\"}",
            "Thought: I have the listing now.\n"
            "Final Answer: I listed the current directory above.",
        ]
    )


def build_agent(args) -> Agent:
    config = Config()
    if args.model:
        config.model_path = args.model
    if args.workdir:
        config.workdir = args.workdir
    if args.no_shell:
        config.allow_shell = False
    if args.no_think:
        config.no_think = True
    config.verbose = not args.quiet

    registry = build_default_registry(config)

    if args.mock:
        engine = _demo_mock_engine()
    elif args.server:
        engine = LlamaServerEngine(base_url=args.server)
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
            "  --server http://localhost:8080   (recommended; talk to a llama.cpp server)\n"
            "  --model path/to/model.gguf       (in-process, needs a compatible wheel)\n"
            "  --mock                           (no model; exercise the agent loop)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return Agent(engine=engine, tools=registry, config=config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cup", description="Agentic AI for weak/legacy CPUs.")
    parser.add_argument("--version", action="version", version=f"cup {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the agent on a task")
    run.add_argument("task", help="What you want the agent to do")
    run.add_argument("--model", help="Path to a .gguf model (or set CUP_MODEL)")
    run.add_argument(
        "--server",
        help="URL of a running llama.cpp server, e.g. http://localhost:8080 "
        "(recommended on legacy CPUs without AVX-512)",
    )
    run.add_argument("--workdir", help="Directory the agent is confined to (default: CWD)")
    run.add_argument("--mock", action="store_true", help="Use the scripted mock engine")
    run.add_argument("--no-shell", action="store_true", help="Disable the run_command tool")
    run.add_argument(
        "--no-think", action="store_true",
        help="Disable a reasoning model's thinking phase (recommended for Qwen3)",
    )
    run.add_argument("--quiet", action="store_true", help="Don't stream reasoning")

    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 0

    agent = build_agent(args)
    print(f"\n> Task: {args.task}\n" + "-" * 60)
    result = agent.run(args.task)
    print("-" * 60)
    print(f"\nAnswer ({result.stopped_reason}, {len(result.steps)} steps):\n{result.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
