"""Create deterministic, balanced train/validation/test subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .common import DATASET_ID, DATASET_REVISION, write_jsonl


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def pick_balanced(
    train_rows: list[dict],
    test_rows: list[dict],
    label_count: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Keep official test separate and remove exact normalized text overlap."""
    if min(train_per_class, val_per_class, test_per_class) < 1:
        raise ValueError("All per-class counts must be positive")
    by_train: dict[int, list[dict]] = defaultdict(list)
    by_test: dict[int, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_train[row["label_id"]].append(row)
    for row in test_rows:
        by_test[row["label_id"]].append(row)

    chosen = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    shuffled_train = {}
    for label_id in range(label_count):
        candidates = by_train[label_id][:]
        random.Random(seed + label_id).shuffle(candidates)
        shuffled_train[label_id] = candidates
    # Select all validation rows first so a larger training subset does not
    # change the validation set used to compare training runs.
    for split, required in (("validation", val_per_class), ("train", train_per_class)):
        for label_id in range(label_count):
            count = 0
            for row in shuffled_train[label_id]:
                key = normalized(row["text"])
                if not key or key in seen:
                    continue
                chosen[split].append(row)
                seen.add(key)
                count += 1
                if count == required:
                    break
            if count != required:
                raise ValueError(f"Not enough unique {split} rows for label {label_id}")

    # Exclude every official-train text, not just selected rows. This keeps
    # the held-out sample fixed when train_per_class changes.
    official_train_texts = {normalized(row["text"]) for row in train_rows}
    for label_id in range(label_count):
        candidates = by_test[label_id][:]
        random.Random(seed + 10000 + label_id).shuffle(candidates)
        count = 0
        for row in candidates:
            key = normalized(row["text"])
            if not key or key in seen or key in official_train_texts:
                continue
            chosen["test"].append(row)
            seen.add(key)
            count += 1
            if count == test_per_class:
                break
        if count != test_per_class:
            raise ValueError(f"Not enough unique test rows for label {label_id}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=3)
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    labels = dataset["train"].features["label"].names
    if len(labels) != 77 or dataset["test"].features["label"].names != labels:
        raise ValueError("Unexpected BANKING77 label schema")

    def as_rows(split: str) -> list[dict]:
        return [
            {"text": row["text"], "label_id": row["label"], "label": labels[row["label"]]}
            for row in dataset[split]
        ]

    chosen = pick_balanced(
        as_rows("train"), as_rows("test"), len(labels),
        args.train_per_class, args.val_per_class, args.test_per_class, args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in chosen.items():
        path = args.out / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_splits": {"train": "official train", "validation": "official train", "test": "official test"},
        "seed": args.seed,
        "labels": labels,
        "counts": {split: len(rows) for split, rows in chosen.items()},
        "sha256": hashes,
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": metadata["counts"], "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
