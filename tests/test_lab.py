import unittest

from banking_lab.evaluate import score_rows
from banking_lab.prepare import normalized, pick_balanced
from banking_lab.report import paired_report


class DataSplitTests(unittest.TestCase):
    def test_balanced_deterministic_and_no_text_leakage(self):
        train = [
            {"text": f"Train {label} {i}", "label_id": label, "label": f"intent_{label}"}
            for label in range(2) for i in range(8)
        ]
        test = [
            {"text": f"Test {label} {i}", "label_id": label, "label": f"intent_{label}"}
            for label in range(2) for i in range(5)
        ]
        # This official test row must not leak into the held-out subset.
        test[0]["text"] = train[0]["text"].upper()
        first = pick_balanced(train, test, 2, 3, 2, 2, 42)
        self.assertEqual(first, pick_balanced(train, test, 2, 3, 2, 2, 42))
        self.assertEqual({k: len(v) for k, v in first.items()},
                         {"train": 6, "validation": 4, "test": 4})
        larger = pick_balanced(train, test, 2, 4, 2, 2, 42)
        self.assertEqual(first["validation"], larger["validation"])
        self.assertEqual(first["test"], larger["test"])
        all_text = [normalized(row["text"]) for rows in first.values() for row in rows]
        self.assertEqual(len(all_text), len(set(all_text)))


class ScoringTests(unittest.TestCase):
    def test_invalid_labels_are_separate_from_accuracy(self):
        labels = ["a", "b"]
        rows = [
            {"gold": "a", "predicted": "a"},
            {"gold": "b", "predicted": None},
        ]
        score = score_rows(rows, labels)
        self.assertEqual(score["correct_count"], 1)
        self.assertEqual(score["valid_label_count"], 1)
        self.assertEqual(score["macro_accuracy"], 0.5)

    def test_paired_report_rejects_different_test_rows(self):
        base = [{"index": 0, "text": "x", "gold": "a", "raw": "b",
                 "predicted": "b", "correct": False}]
        tuned = [{"index": 0, "text": "y", "gold": "a", "raw": "a",
                  "predicted": "a", "correct": True}]
        with self.assertRaises(ValueError):
            paired_report(base, tuned)

    def test_paired_improvements_and_regressions(self):
        base = [
            {"index": 0, "text": "x", "gold": "a", "raw": "b",
             "predicted": "b", "correct": False},
            {"index": 1, "text": "y", "gold": "b", "raw": "b",
             "predicted": "b", "correct": True},
        ]
        tuned = [
            {"index": 0, "text": "x", "gold": "a", "raw": "a",
             "predicted": "a", "correct": True},
            {"index": 1, "text": "y", "gold": "b", "raw": "a",
             "predicted": "a", "correct": False},
        ]
        result = paired_report(base, tuned)
        self.assertEqual(result["improved_count"], 1)
        self.assertEqual(result["regressed_count"], 1)
        self.assertEqual(result["accuracy_delta"], 0)


if __name__ == "__main__":
    unittest.main()
