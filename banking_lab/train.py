"""Run one-GPU Gemma 4 QLoRA supervised fine tuning with TRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    MODEL_ID, MODEL_REVISION, TOKENIZER_ID, TOKENIZER_REVISION,
    load_metadata, prompt_for, read_jsonl,
)


def sft_rows(rows: list[dict], labels: list[str]) -> list[dict]:
    return [
        {
            "prompt": prompt_for(row["text"], labels),
            "completion": [{"role": "assistant", "content": row["label"]}],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This lab requires one CUDA GPU with BF16 support")

    metadata = load_metadata(args.data_dir)
    labels = metadata["labels"]
    train = Dataset.from_list(sft_rows(read_jsonl(args.data_dir / "train.jsonl"), labels))
    validation = Dataset.from_list(
        sft_rows(read_jsonl(args.data_dir / "validation.jsonl"), labels)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # PEFT's Gemma 4 default targets language-model layers. We add no
        # special tokens, so large embedding and output matrices stay frozen.
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        logging_dir=str(args.output_dir / "tensorboard"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=768,
        completion_only_loss=True,
        packing=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_torch_fused",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to=["tensorboard"],
        model_init_kwargs={"dtype": "bfloat16", "revision": MODEL_REVISION},
        seed=args.seed,
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=lora,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.4%})")
    result = trainer.train()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.adapter_dir))
    tokenizer.save_pretrained(args.adapter_dir)
    summary = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset_revision": metadata["dataset_revision"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "PEFT Gemma 4 defaults"},
        "quantization": "4-bit NF4, double quantization, BF16 compute",
        "settings": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                     "batch_size": 1, "gradient_accumulation": 8, "max_length": 768},
        "train_metrics": result.metrics,
        "log_history": trainer.state.log_history,
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Adapter saved to {args.adapter_dir}")


if __name__ == "__main__":
    main()
