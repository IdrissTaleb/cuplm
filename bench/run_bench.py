"""Benchmark / eval harness for the Cup agent framework.

Two modes:

  * Mock mode (no model, deterministic):
        python -m bench.run_bench --mock

  * Real model via a running llama.cpp server:
        python -m bench.run_bench --server http://localhost:8080

For every task the harness:
  1. creates a fresh temp workdir (tempfile.mkdtemp),
  2. runs the task's setup to seed files,
  3. runs the agent on the task prompt,
  4. runs the task's check against the workdir + Result,
  5. records pass/fail, wall-clock seconds, step count, and stopped_reason.

It then prints a results table and a summary, and exits 0 iff every task passed.
This harness never starts a server and never executes downloaded binaries; in
`--server` mode it only makes HTTP calls through LlamaServerEngine.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

# Ensure the repo root is importable when run as `python -m bench.run_bench`
# from the repo root (it is, via the package), and also if invoked oddly.
try:
    from cup import Agent, Config, LlamaServerEngine, MockEngine
    from cup.skills import build_default_registry
except ModuleNotFoundError:  # pragma: no cover - fallback for odd invocations
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cup import Agent, Config, LlamaServerEngine, MockEngine
    from cup.skills import build_default_registry

from bench.tasks import TASKS, Task


@dataclass
class TaskOutcome:
    name: str
    passed: bool
    seconds: float
    steps: int
    reason: str
    error: Optional[str] = None


def _build_engine(task: Task, mode: str, server_url: str):
    if mode == "mock":
        return MockEngine(task.mock_responses)
    return LlamaServerEngine(base_url=server_url)


def run_task(task: Task, mode: str, server_url: str) -> TaskOutcome:
    workdir = tempfile.mkdtemp(prefix=f"cupbench_{task.name}_")
    try:
        task.setup(workdir)

        config = Config(
            workdir=workdir,
            verbose=False,
            allow_shell=True,
            max_steps=8,
        )
        registry = build_default_registry(config)
        engine = _build_engine(task, mode, server_url)
        agent = Agent(engine=engine, tools=registry, config=config)

        start = time.perf_counter()
        result = agent.run(task.prompt)
        elapsed = time.perf_counter() - start

        try:
            passed = bool(task.check(workdir, result))
            err = None
        except Exception as exc:  # a buggy check must not abort the whole suite
            passed = False
            err = f"check raised {type(exc).__name__}: {exc}"

        return TaskOutcome(
            name=task.name,
            passed=passed,
            seconds=elapsed,
            steps=len(result.steps),
            reason=result.stopped_reason,
            error=err,
        )
    except Exception as exc:
        # A failure in setup / agent run: record it rather than crash.
        return TaskOutcome(
            name=task.name,
            passed=False,
            seconds=0.0,
            steps=0,
            reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _print_table(outcomes: List[TaskOutcome]) -> None:
    headers = ("name", "pass", "secs", "steps", "reason")
    rows = [
        (
            o.name,
            "PASS" if o.passed else "FAIL",
            f"{o.seconds:.2f}",
            str(o.steps),
            o.reason,
        )
        for o in outcomes
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cols) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    # Surface any errors below the table for debuggability.
    for o in outcomes:
        if o.error:
            print(f"  ! {o.name}: {o.error}")


def _print_summary(outcomes: List[TaskOutcome]) -> None:
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    total_secs = sum(o.seconds for o in outcomes)
    avg_secs = total_secs / total if total else 0.0
    print()
    print(
        f"{passed}/{total} passed, "
        f"avg {avg_secs:.1f}s, total {total_secs:.1f}s"
    )


def _check_server_reachable(server_url: str) -> bool:
    """Probe the server with a trivial completion. Returns False if unreachable."""
    engine = LlamaServerEngine(base_url=server_url, timeout=10.0)
    try:
        engine.generate("ping", stop=["\n"], max_tokens=1, temperature=0.0)
        return True
    except Exception as exc:
        print(f"ERROR: could not reach llama.cpp server at {server_url}: {exc}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.run_bench",
        description="Benchmark harness for the Cup agent framework.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mock",
        action="store_true",
        help="Run every task with a deterministic MockEngine (no model).",
    )
    group.add_argument(
        "--server",
        metavar="URL",
        help="Run every task against a running llama.cpp server (e.g. "
        "http://localhost:8080).",
    )
    args = parser.parse_args(argv)

    if args.mock:
        mode, server_url = "mock", ""
        print("Running Cup benchmark in MOCK mode (no model).\n")
    else:
        mode, server_url = "server", args.server
        print(f"Running Cup benchmark against server {server_url}.\n")
        if not _check_server_reachable(server_url):
            return 2

    outcomes = [run_task(task, mode, server_url) for task in TASKS]

    _print_table(outcomes)
    _print_summary(outcomes)

    all_passed = all(o.passed for o in outcomes)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
