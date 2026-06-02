# Training `cuplm`

This directory will hold the pipeline that turns a permissively-licensed open base
model into **`cuplm`** — a tiny model specialized for reliable tool-calling on weak
CPUs. (Not started yet; this is the Phase 2 plan.)

## Strategy (decided)

We do **not** pretrain from scratch. A from-scratch sub-1B model trained on a
realistic budget would be far weaker than existing tiny models and unable to do
agentic tasks. Instead we **fine-tune an open base** — the value is in the data and
the specialization, not in reinventing pretraining.

**Base candidates** (both permissive — Apache-2.0 / MIT, so we can rebrand and ship):

| Base | Params | Why |
|------|--------|-----|
| Qwen3-0.6B | 0.6B | Strong small-model tool-calling baseline |
| SmolLM2-360M | 360M | Smaller; better fit for the 1–2GB tier later |

## Pipeline (planned)

1. **Collect tasks** — file/system automation prompts matching Cup's skills.
2. **Distill** — generate high-quality ReAct traces (Thought/Action/Action Input/
   Observation/Final Answer) from a strong teacher model; filter for traces that
   actually succeed when replayed through Cup's tools.
3. **Format** — emit `data/*.jsonl` in the exact prompt format from `cup/prompts.py`
   so training matches inference.
4. **Fine-tune** — QLoRA on a single consumer/cloud GPU (est. \$50–500).
5. **Quantize** — export to GGUF (Q4_K_M) with `llama-quantize`.
6. **Evaluate** — run the Phase 3 task benchmark on a 4GB-class machine and report
   task success rate vs. the stock base.

## Success criterion

`cuplm-v0` should beat its own base model at Cup's file/system tasks on a 4GB
laptop — that's what justifies it being a distinct, named model rather than "just
use the base."
