# Cup benchmark harness

A small eval harness for the **Cup** agent framework. It runs a fixed set of
local file/system-automation tasks through the agent loop, scores each one, and
reports timing and reliability.

## What it does

For every task in `bench/tasks.py` the harness:

1. creates a fresh temporary working directory,
2. runs the task's `setup` to seed files,
3. runs the agent on the task's prompt,
4. runs the task's `check` against the workdir and the agent `Result`,
5. records pass/fail, wall-clock seconds, number of steps, and the stop reason.

Checks are intentionally tolerant (case-insensitive substring matches and
integer extraction) because tiny models phrase answers loosely.

## Running

Run from the repository root (`C:\Users\realm\cuplm`).

### Mock mode (no model, deterministic)

```
python -m bench.run_bench --mock
```

Each task carries scripted `MockEngine` responses, so this exercises the entire
harness — parsing, tool dispatch, observation injection, checks, timing — with
no model and no network. Use it to verify the harness itself. All mock tasks are
expected to pass.

### Server mode (real model)

Start a llama.cpp server yourself first, e.g.:

```
llama-server -m model.gguf -c 4096 -t 4 --port 8080
```

Then point the harness at it:

```
python -m bench.run_bench --server http://localhost:8080
```

The harness only makes HTTP calls via `LlamaServerEngine`; it never starts a
server or runs any binary. If the server is unreachable it prints a clear
message and exits with code **2**.

## Interpreting the output

The table has one row per task:

| column | meaning |
| ------ | ------- |
| name   | task identifier |
| pass   | `PASS` if the check succeeded, else `FAIL` |
| secs   | wall-clock seconds for the agent run (`time.perf_counter`) |
| steps  | number of think/act/observe cycles the agent took |
| reason | `final` (clean finish), `max_steps` (hit the cap), or `stuck` (couldn't follow the format) |

Any per-task error (failed setup, exception in a check, etc.) is printed below
the table prefixed with `!`.

The summary line reports `X/Y passed, avg Z.Zs, total T.Ts`.

## Exit codes

| code | meaning |
| ---- | ------- |
| 0    | every task passed |
| 1    | at least one task failed |
| 2    | (`--server` only) the server was unreachable |
