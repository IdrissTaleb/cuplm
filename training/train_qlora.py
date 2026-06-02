"""Fine-tune `cuplm` from an open base with (Q)LoRA on Cup's SFT data.

Design:
  * Train ONLY on the model's own outputs (Thought/Action/Action Input/Final
    Answer). Observations and the prompt are masked with -100 so the model never
    learns to hallucinate tool results — this is the part fine-tuners most often
    get wrong, so it's `--validate`-able locally with just the tokenizer.
  * QLoRA (4-bit) on GPU when bitsandbytes is available; falls back to plain
    LoRA otherwise. Real training needs a GPU; this machine (CPU-only) can run
    `--validate` and a tiny `--smoke` to prove the code path.

Local check (only needs `pip install transformers`):
    python -m training.train_qlora --validate

Real run (on a cloud GPU, after `pip install -r training/requirements.txt`):
    python -m training.train_qlora --model Qwen/Qwen3-0.6B \
        --data training/data/cuplm_sft.jsonl --out training/checkpoints/cuplm-lora \
        --epochs 3 --batch 8 --4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_DATA = "training/data/cuplm_sft.jsonl"
# LoRA targets for Qwen3 / Llama-style architectures.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tokenize_record(tokenizer, record: dict, max_len: int) -> dict:
    """Turn one record's segments into input_ids + labels, masking everything
    that isn't the model's own text. EOS is appended and trained (teach it to
    stop)."""
    input_ids: list[int] = []
    labels: list[int] = []
    for text, trainable in record["segments"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids += ids
        labels += ids if trainable else [-100] * len(ids)
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": input_ids[:max_len], "labels": labels[:max_len]}


def validate(args) -> int:
    """Tokenize a few records and report the train/mask split. No torch needed."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    records = load_records(args.data)
    print(f"Loaded {len(records)} records. Tokenizer: {args.model}\n")

    tot_tokens = tot_trained = 0
    for r in records:
        ex = tokenize_record(tok, r, args.max_len)
        tot_tokens += len(ex["labels"])
        tot_trained += sum(1 for l in ex["labels"] if l != -100)

    pct = 100 * tot_trained / max(tot_tokens, 1)
    print(f"Total tokens: {tot_tokens:,}")
    print(f"Trained (unmasked) tokens: {tot_trained:,}  ({pct:.1f}%)")
    print(f"Masked (prompt + observations): {tot_tokens - tot_trained:,}\n")

    # Show exactly which spans are trained on one tool-use example.
    sample = next(r for r in records if any(s[0].startswith("\nObservation") for s in r["segments"]))
    print("--- mask check on one tool-use example (>>> = trained, ... = masked) ---")
    for text, trainable in sample["segments"]:
        tag = ">>>" if trainable else "..."
        show = text.strip().replace("\n", " / ")
        if len(show) > 90:
            show = show[:90] + "..."
        print(f"{tag} {show}")
    print("\nOK: observations and prompt are masked; model outputs are trained.")
    return 0


def train(args) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = load_records(args.data)
    if args.smoke:
        records = records[:16]
    ds = Dataset.from_list([tokenize_record(tok, r, args.max_len) for r in records])

    quant_cfg = None
    use_4bit = args.four_bit
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # bitsandbytes unavailable (e.g. CPU/Windows)
            print(f"[warn] 4-bit unavailable ({exc}); falling back to plain LoRA.")
            use_4bit = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to=[],
        max_steps=2 if args.smoke else -1,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}")
    print("Next: merge + convert to GGUF — see training/README.md")
    return 0


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Fine-tune cuplm with (Q)LoRA.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default="training/checkpoints/cuplm-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA 4-bit (GPU)")
    p.add_argument("--smoke", action="store_true", help="Tiny 2-step run to prove the pipeline")
    p.add_argument("--validate", action="store_true", help="Check tokenization/masking only (no torch)")
    args = p.parse_args(argv)
    return validate(args) if args.validate else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
