"""Shared, versioned inputs and prompt format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

MODEL_ID = "google/gemma-4-E4B"
MODEL_REVISION = "411aa17b749aa952df1359d2dcea73917a544d9a"
TOKENIZER_ID = "google/gemma-4-E4B-it"
TOKENIZER_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"


def prompt_for(text: str, labels: list[str]) -> list[dict[str, str]]:
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Expected the 77 distinct BANKING77 intent names")
    return [
        {
            "role": "system",
            "content": (
                "Classify the customer's banking message. Reply with exactly one intent "
                "name from this list, in lowercase with underscores. Do not explain.\n"
                + ", ".join(labels)
            ),
        },
        {"role": "user", "content": text},
    ]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_metadata(data_dir: Path) -> dict:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Dataset revision does not match this lab")
    if len(metadata["labels"]) != 77:
        raise ValueError("Expected all 77 BANKING77 labels")
    return metadata
