"""Generate deterministic labels from the frozen base or trained adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    MODEL_ID, MODEL_REVISION, TOKENIZER_ID, TOKENIZER_REVISION,
    load_metadata, prompt_for, read_jsonl, write_jsonl,
)


def score_rows(rows: list[dict], labels: list[str]) -> dict:
    if not rows:
        raise ValueError("No predictions to score")
    valid = sum(row["predicted"] in labels for row in rows)
    correct = sum(row["predicted"] == row["gold"] for row in rows)
    by_label = {
        label: [row for row in rows if row["gold"] == label] for label in labels
    }
    macro = sum(
        sum(row["predicted"] == label for row in group) / len(group)
        for label, group in by_label.items() if group
    ) / sum(bool(group) for group in by_label.values())
    return {
        "n": len(rows),
        "valid_label_count": valid,
        "valid_label_rate": valid / len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "macro_accuracy": macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA GPU")
    metadata = load_metadata(args.data_dir)
    labels = metadata["labels"]
    rows = read_jsonl(args.data_dir / f"{args.split}.jsonl")
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
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    if turn_end == tokenizer.unk_token_id:
        turn_end = tokenizer.eos_token_id

    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = tokenizer.apply_chat_template(
                prompt_for(row["text"], labels),
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=turn_end,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = output[0][batch["input_ids"].shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predicted = raw if raw in labels else None
            predictions.append({
                "index": index,
                "text": row["text"],
                "gold": row["label"],
                "raw": raw,
                "predicted": predicted,
                "correct": predicted == row["label"],
            })
            if (index + 1) % 50 == 0:
                print(f"Evaluated {index + 1}/{len(rows)}", flush=True)

    write_jsonl(args.output, predictions)
    summary = {
        "variant": "adapter" if args.adapter_dir else "base",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "split": args.split,
        **score_rows(predictions, labels),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
