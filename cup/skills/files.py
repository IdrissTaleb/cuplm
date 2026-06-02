"""File skills: read, write, list, and search within a confined workdir.

All paths are resolved relative to (and confined within) config.workdir, so the
agent can't wander off into the rest of the filesystem.
"""

from __future__ import annotations

import os
from typing import Union

from cup.tools import Tool

_MAX_READ_BYTES = 20_000  # keep observations small for tiny context windows


def _safe_path(workdir: str, rel: str) -> str:
    """Resolve `rel` under `workdir`, refusing escapes via .. or absolute paths."""
    workdir_abs = os.path.realpath(workdir)
    target = os.path.realpath(os.path.join(workdir_abs, rel))
    if target != workdir_abs and not target.startswith(workdir_abs + os.sep):
        raise ValueError(f"path '{rel}' escapes the working directory")
    return target


def _get(args: Union[dict, str], key: str, default: str = "") -> str:
    if isinstance(args, dict):
        return str(args.get(key, default))
    # Bare string => treat it as the primary positional arg.
    return args if default == "" else default


def make_file_tools(workdir: str):
    def read_file(args):
        path = _get(args, "path")
        if not path:
            return "ERROR: 'path' is required."
        full = _safe_path(workdir, path)
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(_MAX_READ_BYTES + 1)
        if len(data) > _MAX_READ_BYTES:
            return data[:_MAX_READ_BYTES] + "\n...[truncated]"
        return data if data else "(empty file)"

    def write_file(args):
        if not isinstance(args, dict):
            return 'ERROR: write_file needs JSON like {"path": "...", "content": "..."}'
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not path:
            return "ERROR: 'path' is required."
        full = _safe_path(workdir, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Wrote {len(content)} chars to {path}."

    def file_stats(args):
        # Tiny models miscount; compute it deterministically so they don't have to.
        path = _get(args, "path")
        if not path:
            return "ERROR: 'path' is required."
        full = _safe_path(workdir, path)
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        words = len(text.split())
        return f"lines={lines} words={words} chars={len(text)} bytes={os.path.getsize(full)}"

    def list_dir(args):
        path = _get(args, "path", ".") or "."
        full = _safe_path(workdir, path)
        if not os.path.isdir(full):
            return f"ERROR: not a directory: {path}"
        entries = sorted(os.listdir(full))
        if not entries:
            return "(empty directory)"
        out = []
        for name in entries[:200]:
            kind = "dir " if os.path.isdir(os.path.join(full, name)) else "file"
            out.append(f"{kind}  {name}")
        return "\n".join(out)

    return [
        Tool(
            name="read_file",
            description="Read a text file's contents",
            args_hint='{"path": "relative/path.txt"}',
            func=read_file,
        ),
        Tool(
            name="write_file",
            description="Write text to a file (creates/overwrites)",
            args_hint='{"path": "out.txt", "content": "..."}',
            func=write_file,
        ),
        Tool(
            name="list_dir",
            description="List files and folders in a directory",
            args_hint='{"path": "."}',
            func=list_dir,
        ),
        Tool(
            name="file_stats",
            description="Count lines, words, chars and bytes of a file (use this instead of counting yourself)",
            args_hint='{"path": "notes.txt"}',
            func=file_stats,
        ),
    ]
