"""Shell skill: run a command in the workdir, with a safety deny-list.

This is intentionally conservative. Tiny models hallucinate; an agent that can
run arbitrary shell commands is powerful and dangerous. We:
  * confine the working directory,
  * enforce a timeout,
  * block an obvious deny-list of destructive patterns, and
  * cap output size.

This is a *guard*, not a sandbox. For untrusted use, run Cup in a container/VM.
"""

from __future__ import annotations

import re
import subprocess
from typing import Union

from cup.tools import Tool

# Patterns we refuse outright. Not exhaustive — defense in depth, not a sandbox.
_DENY = [
    r"\brm\s+-rf\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",            # fork bomb
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\b",
    r"Remove-Item.*-Recurse.*-Force",
    r"\b>\s*/dev/sd",
]
_DENY_RE = [re.compile(p, re.IGNORECASE) for p in _DENY]

_MAX_OUTPUT = 8_000


def _cmd_from(args: Union[dict, str]) -> str:
    if isinstance(args, dict):
        return str(args.get("command", args.get("cmd", "")))
    return str(args)


def make_shell_tool(workdir: str, timeout: int = 30):
    def run_command(args):
        command = _cmd_from(args).strip()
        if not command:
            return "ERROR: 'command' is required."
        for pat in _DENY_RE:
            if pat.search(command):
                return f"REFUSED: command matches a blocked pattern ({pat.pattern})."
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s."
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.strip() or "(no output)"
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n...[truncated]"
        return f"[exit {proc.returncode}]\n{out}"

    return Tool(
        name="run_command",
        description="Run a shell command in the working directory",
        args_hint='{"command": "ls -la"}',
        func=run_command,
    )
