# Walkthrough: what does an adapter learn?

**Goal:** By the end, explain SFT, completion-only loss, LoRA, 4-bit quantization, validation versus test, and how to decide whether the adapter helped.

## 1. Inspect the task

Read [BANKING77's dataset card](https://huggingface.co/datasets/PolyAI/banking77). It has 13,083 English customer queries and 77 fine-grained intents: 10,003 official train and 3,080 official test rows. Compare `card_arrival`, `card_delivery_estimate`, and `card_not_working`: these near-neighbor labels make exact-match accuracy more informative than a trivial keyword task.

Open [prepare.py](banking_lab/prepare.py). Record the seed, per-class counts, split source, and normalized duplicate rule. The generated `/work/data/metadata.json` includes a SHA-256 for each JSONL file. Validation is selected only from official training data; the official test set is never passed to the trainer.

**Question:** Why is a random sample from all 13,083 records an invalid evaluation split?

## 2. Inspect the training examples

Open [common.py](banking_lab/common.py) and [train.py](banking_lab/train.py). The prompt contains the same instruction at training and evaluation time. A completion is only the intent name. TRL's conversational prompt-completion representation applies the Gemma chat template and masks prompt tokens from the loss.

**Question:** Which tokens receive gradient updates? What would change if the prompt tokens were included in the loss?

## 3. Train on one H200

Use [README.md](README.md#run-on-ocpai-02) to build and submit the Job. It runs:

```text
prepare → frozen base inference → TRL SFT + PEFT QLoRA → adapter inference → paired report
```

Inspect [train.py](banking_lab/train.py). NF4 stores the base in four bits; the base remains frozen. LoRA adds two trainable low-rank matrices to selected language-model layers. The script prints trainable versus total parameter counts. It uses BF16 math and gradient checkpointing on the H200. The adapter is saved separately from the base model.

| Setting | Teaching default | Why it is visible |
|---|---:|---|
| Training rows | 8 per intent, 616 total | Small-data effect |
| Validation rows | 3 per intent, 231 total | Monitor overfitting |
| Epochs | 2 | Observe loss across passes |
| Microbatch / accumulation | 1 / 8 | Effective batch size 8 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | Adapter capacity and regularization |
| Quantization | 4-bit NF4 + double quant | Reduce frozen-base memory |
| Learning rate / schedule | 2e-4 / cosine | Size and decay of updates |
| Max sequence length | 768 | Includes the 77-label instruction; longest sampled example measured 650 tokens |
| Test rows | 5 per intent, 385 total | Fast, balanced comparison |

**Question:** Why does reducing the base weights to four bits not mean the gradient computation is four-bit?

## 4. Interpret the evidence

Follow the Job log and open TensorBoard as described in the README. Compare training loss with validation loss at each epoch. The [report](banking_lab/report.py) gives:

- **Valid label rate:** the complete generated text is exactly one of the 77 label names.
- **Exact intent accuracy:** that valid name matches the dataset label.
- **Improved/regressed cases:** individual messages fixed or broken by the adapter.

Record your observed result here after a real run:

| Held-out measure | Base | Adapter |
|---|---:|---:|
| Valid label names / 385 | — | — |
| Correct intents / 385 | — | — |

The report is generated from the same 385 records for both variants. Do not infer production readiness from one sample, one seed, or training loss alone. If accuracy falls, inspect wrong cases and validation loss before changing any setting. A single 77-class label can be ambiguous; errors may reflect the dataset taxonomy as well as the model.

**Question:** Does the adapter improve classification, merely improve response formatting, both, or neither?

## 5. Optional extension

For a second teaching run, change only `--train-per-class 8` to `20` in the Job and use a new Job name and artifact PVC. Keep the validation/test selection seed and test rows fixed, and compare the two adapters. Use the validation split to choose settings; reserve the official test subset for the final comparison. For an official benchmark, evaluate all 3,080 test rows and report the full setup separately.

The [Google Gemma guide](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora), [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer), and [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) provide the upstream mechanics; this lab changes their example task to fine-grained intent classification.
