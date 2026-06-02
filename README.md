<div align="center">

# Cup ☕ + cuplm

**Agentic AI that runs on the computer you already own — even an old one.**

Cup is a tiny, dependency-light agent framework that drives small language models
(`cuplm`) on plain CPUs. No GPU. No cloud. No subscription. It targets weak and
legacy hardware — the headline goal is a capable agent running *fully offline on a
laptop with ~4GB of RAM*.

[Why](#why) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Roadmap](#roadmap)

</div>

---

## Why

Agentic AI is mostly locked behind expensive GPUs and paid cloud APIs. But the
inference problem on CPUs is *solved* ([llama.cpp](https://github.com/ggml-org/llama.cpp)),
and small models keep getting better. The missing piece is a **batteries-included
agent framework tuned for the failure modes of tiny models on weak hardware**.

That's Cup. The mission: make agentic AI affordable and accessible to anyone with
an old device.

## Status

**Phase 0 — early alpha.** The agent loop, tool system, file/system skills, and CLI
are built and covered by tests that run the whole loop end-to-end with no model
download. Real inference works through a local llama.cpp server (see below).
`cuplm` (our fine-tuned tiny model) is not trained yet — Cup currently runs any
GGUF model.

## Quickstart

### Option A — talk to a llama.cpp server (recommended, works on any CPU)

The official llama.cpp binaries do **runtime CPU detection**, so they run on older
chips without AVX-512 (where many prebuilt Python wheels crash). Grab a build from
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), then:

```bash
# 1. start the server with any GGUF model
llama-server -m models/your-model.gguf -c 4096 -t 4 --port 8080

# 2. install Cup
pip install -e .

# 3. run an agent task against it
cup run "Read notes.txt and tell me how many lines it has." \
    --server http://localhost:8080 \
    --workdir ./demo
```

### Option B — in-process (needs a wheel built for your CPU)

```bash
pip install -e ".[llama]"
cup run "list the files here" --model models/your-model.gguf --workdir ./demo
```

> ⚠️ On CPUs without AVX-512, the generic `llama-cpp-python` CPU wheel may crash
> with an illegal instruction (`0xc000001d` on Windows). Use Option A, or build the
> wheel from source for your CPU. This portability gap is exactly the problem Cup
> exists to smooth over.

### Option C — no model at all (try the agent loop)

```bash
cup run "list files and summarize" --mock
```

## How it works

```
your task ──▶ prompt (few-shot, tiny-model-friendly)
                 │
                 ▼
            Engine.generate ──▶ [ llama.cpp server | in-process | mock ]
                 │
                 ▼
        tolerant ReAct parser ──▶ Thought / Action / Action Input
                 │                         │
          Final Answer?                run tool (file/shell skill)
                 │                         │
                 ▼                         ▼
              return ◀──────────── inject Observation, loop
```

The hard part of running agents on a 0.5B model isn't the loop — it's *reliability*.
Cup's design choices target that directly:

- **Tolerant parsing.** Tiny models mangle JSON. The parser accepts messy output,
  strips code fences, falls back to plain strings, and recovers from unknown tools
  instead of crashing. (`cup/tools.py`)
- **Few-shot, short system prompt.** Small models pattern-match better than they
  follow instructions. (`cup/prompts.py`)
- **A boring, debuggable loop** with a hard step cap so it never hangs. (`cup/agent.py`)
- **Confined, guarded skills.** File access is sandboxed to a working directory;
  the shell skill has a destructive-command deny-list. (`cup/skills/`)

## Skills (built in)

| Tool | Does |
|------|------|
| `read_file` | Read a text file (confined to the workdir) |
| `write_file` | Create/overwrite a file |
| `list_dir` | List a directory |
| `run_command` | Run a shell command (deny-list + timeout; toggle with `--no-shell`) |

## Roadmap

- **Phase 1** — harden reliability on a real tiny base (Qwen3-0.6B / SmolLM2-360M).
- **Phase 2** — train **`cuplm`**: distill a tool-calling dataset, QLoRA fine-tune,
  quantize to GGUF. See [`training/`](training/).
- **Phase 3** — GBNF grammar-constrained decoding to force valid tool calls; a task
  benchmark to report success rate on a 4GB laptop.
- **Phase 4** — packaging, demo, launch.
- **Phase 5** — optional premium layer (hosted fine-tuning, skills marketplace) on
  top of the open core.

## License

[Apache-2.0](LICENSE).
