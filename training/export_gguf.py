"""Merge a trained LoRA adapter into the base model and prep for GGUF export.

GGUF conversion + quantization is done by llama.cpp tooling (the quantizer is
already vendored at vendor/llama-cpu/llama-quantize.exe). This script handles
the PEFT-specific part — merging the adapter into full weights — then prints the
exact convert/quantize commands.

Usage (on the box where you trained):
    python -m training.export_gguf --base Qwen/Qwen3-0.6B \
        --adapter training/checkpoints/cuplm-lora --out training/merged/cuplm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Merge LoRA adapter -> full weights for GGUF export.")
    p.add_argument("--base", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default="training/checkpoints/cuplm-lora")
    p.add_argument("--out", default="training/merged/cuplm")
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {args.base} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
    print(f"Applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging ...")
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f"\nMerged model written to {out}\n")

    print("Next — convert to GGUF and quantize (needs llama.cpp's convert script):")
    print("  1) pip install gguf")
    print("  2) python path/to/llama.cpp/convert_hf_to_gguf.py "
          f"{out} --outfile training/merged/cuplm-f16.gguf --outtype f16")
    print("  3) vendor\\llama-cpu\\llama-quantize.exe training/merged/cuplm-f16.gguf "
          "models/cuplm-v0-Q4_K_M.gguf Q4_K_M")
    print("\nThen run it through Cup:")
    print("  llama-server -m models/cuplm-v0-Q4_K_M.gguf -c 4096 -t 2 --port 8080")
    print("  cup chat --no-think")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
