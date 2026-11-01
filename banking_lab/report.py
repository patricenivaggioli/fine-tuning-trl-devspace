"""Compare base and tuned predictions on exactly the same held-out records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl


def paired_report(base: list[dict], tuned: list[dict]) -> dict:
    if not base or len(base) != len(tuned):
        raise ValueError("Both prediction files need the same nonzero row count")
    for left, right in zip(base, tuned):
        if (left["index"], left["text"], left["gold"]) != (
            right["index"], right["text"], right["gold"]
        ):
            raise ValueError("Prediction files do not refer to the same test rows")
    n = len(base)
    b_correct = sum(row["correct"] for row in base)
    t_correct = sum(row["correct"] for row in tuned)
    b_valid = sum(row["predicted"] is not None for row in base)
    t_valid = sum(row["predicted"] is not None for row in tuned)
    improved = [
        {"text": b["text"], "gold": b["gold"], "base": b["raw"], "tuned": t["raw"]}
        for b, t in zip(base, tuned) if not b["correct"] and t["correct"]
    ]
    regressed = [
        {"text": b["text"], "gold": b["gold"], "base": b["raw"], "tuned": t["raw"]}
        for b, t in zip(base, tuned) if b["correct"] and not t["correct"]
    ]
    return {
        "n": n,
        "base": {"correct": b_correct, "accuracy": b_correct / n,
                 "valid_labels": b_valid, "valid_label_rate": b_valid / n},
        "tuned": {"correct": t_correct, "accuracy": t_correct / n,
                  "valid_labels": t_valid, "valid_label_rate": t_valid / n},
        "accuracy_delta": (t_correct - b_correct) / n,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "improved_examples": improved[:5],
        "regressed_examples": regressed[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output prefix, without extension")
    args = parser.parse_args()
    result = paired_report(read_jsonl(args.base), read_jsonl(args.tuned))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = (
        "# BANKING77 held-out comparison\n\n"
        f"Samples: {result['n']}\n\n"
        "| Metric | Base | Tuned |\n|---|---:|---:|\n"
        f"| Exact intent accuracy | {result['base']['correct']}/{result['n']} "
        f"({result['base']['accuracy']:.1%}) | {result['tuned']['correct']}/{result['n']} "
        f"({result['tuned']['accuracy']:.1%}) |\n"
        f"| Valid intent name | {result['base']['valid_labels']}/{result['n']} "
        f"({result['base']['valid_label_rate']:.1%}) | "
        f"{result['tuned']['valid_labels']}/{result['n']} "
        f"({result['tuned']['valid_label_rate']:.1%}) |\n\n"
        f"Improved cases: {result['improved_count']}; regressed cases: "
        f"{result['regressed_count']}.\n\n"
        "These are descriptive results on a small, balanced teaching subset, "
        "not a production quality claim.\n"
    )
    args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
