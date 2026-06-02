# Training `cuplm`

Turns a permissively-licensed open base into **`cuplm`** — a tiny model
specialized for reliable tool-calling + direct answers on weak CPUs.

We do **not** pretrain from scratch (a from-scratch sub-1B model would be far
weaker and can't do agentic tasks). We **fine-tune an open base**; the value is
in the data and the specialization.

**Base:** `Qwen/Qwen3-0.6B` (Apache-2.0). Fits the 4GB target at Q4 and is a
strong small-model tool-caller. (`SmolLM2-360M` is a smaller alternative for the
1–2GB tier later.)

## The pipeline (built, validated locally)

```
generate_data.py ──▶ data/cuplm_sft.jsonl ──▶ train_qlora.py ──▶ LoRA adapter
   (self-verifying)                              (QLoRA on GPU)        │
                                                                       ▼
                              cuplm-v0-Q4.gguf ◀── export_gguf.py + llama.cpp
```

### 1. Generate data (runs anywhere, no GPU)
```bash
python -m training.generate_data
```
Produces 580 examples: **tool-use trajectories** whose observations come from
running Cup's *real* tools (so they're correct by construction), plus
**direct-answer examples** (identity, math, how-to) that teach the model NOT to
reach for a tool on general questions — fixing the two weaknesses we saw on
stock Qwen3-0.6B. Format is identical to what Cup sends at inference.

### 2. Validate the label masking (no GPU, only needs `transformers`)
```bash
python -m training.train_qlora --validate
```
Confirms we train **only** on the model's own output (Thought/Action/Final
Answer) and mask the prompt + observations. (Also covered by `tests/test_training.py`.)

### 3. Fine-tune (needs a GPU — e.g. Colab T4, RunPod)
```bash
pip install -r training/requirements.txt
python -m training.train_qlora \
    --model Qwen/Qwen3-0.6B \
    --data training/data/cuplm_sft.jsonl \
    --out training/checkpoints/cuplm-lora \
    --epochs 3 --batch 8 --4bit
```
`--4bit` enables QLoRA (GPU). On CPU/Windows it auto-falls back to plain LoRA.
Estimated cost on a rented T4/A10: a few dollars. A 2-step `--smoke` run proves
the pipeline before you commit to a full run.

### 4. Export to GGUF
```bash
python -m training.export_gguf --base Qwen/Qwen3-0.6B \
    --adapter training/checkpoints/cuplm-lora --out training/merged/cuplm
# then convert + quantize (the script prints the exact commands):
python path/to/llama.cpp/convert_hf_to_gguf.py training/merged/cuplm \
    --outfile training/merged/cuplm-f16.gguf --outtype f16
vendor/llama-cpu/llama-quantize.exe training/merged/cuplm-f16.gguf \
    models/cuplm-v0-Q4_K_M.gguf Q4_K_M
```

### 5. Run it
```bash
llama-server -m models/cuplm-v0-Q4_K_M.gguf -c 4096 -t 2 --port 8080
cup chat --no-think
```

## Success criterion
`cuplm-v0` should beat stock Qwen3-0.6B on the Cup benchmark
(`python -m bench.run_bench --server …`): higher task success, fewer wasted
tool calls, and a reliable identity/direct-answer.
