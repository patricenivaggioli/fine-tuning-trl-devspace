# Gemma 4 + TRL + PEFT: BANKING77 lab

This repository is a teaching lab for **supervised fine tuning (SFT)**. It adapts the text-only path in [Google's Gemma 4 QLoRA walkthrough](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora) to [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77): classify a customer query into one of 77 banking intents. It compares the same frozen base model before and after adding a trained LoRA adapter. No result is claimed until the Job runs.

## What learners should see

1. **Data and split discipline.** The official BANKING77 train set supplies a balanced training subset and a separate validation subset. Only the official test set supplies held-out examples. Exact normalized text overlaps are removed.
2. **Prompt and target.** Each example contains the same 77-label instruction and customer message. The target completion is exactly one label, such as `card_arrival`. TRL computes loss on the assistant completion, not on the prompt.
3. **Parameter-efficient training.** The [Gemma 4 E4B base model](https://huggingface.co/google/gemma-4-E4B) is frozen in 4-bit NF4 form. PEFT trains rank-16 LoRA matrices in its language-model layers. The [E4B instruction tokenizer](https://huggingface.co/google/gemma-4-E4B-it) supplies Gemma 4's conversation template; no vocabulary tokens are added. “E4B” means about 4.5B effective parameters, but about 8B total parameters including embeddings.
4. **Measured change.** The base and adapter run deterministic generation on identical test messages. The report separates exact intent accuracy from the rate of outputs that are valid intent names, and shows both improved and regressed cases.

The default sample is **616 train / 231 validation / 385 test** examples (8 / 3 / 5 per intent). Two epochs, learning rate `2e-4`, effective batch size 8, max length 768, BF16 compute, and one H200 NVL are starting settings, not tuned claims. Training loss and validation loss are written to TensorBoard. See [LAB.md](LAB.md) for the exercise and interpretation.

## Run on OCPAI-02

Open [OpenShift Dev Spaces](https://devspaces.apps.ocpai-02.emea-geo-lab.com/) through the lab VPN and import a **hosted copy** of this repository. Its [devfile](.devfile.yaml) starts a CPU development workspace; the H200 is requested by a separate OpenShift Job. The attached local Git repository currently has no remote, so publish it to your chosen Git service before importing by URL.

In the Dev Spaces terminal, select a project with permission to build images, create an 80 GiB PVC, and request one full H200:

```sh
oc whoami
oc project YOUR_GPU_PROJECT
bash scripts/build-and-run.sh
oc logs -f job/gemma4-banking77
oc get job gemma4-banking77
```

The script builds the image using the repository [Dockerfile](Dockerfile), sending only its Dockerfile, dependency list, and Python source as the build context. It then creates the artifact PVC and submits the [GPU Job](deploy/train-job.yaml). It refuses to replace an existing Job. Its five sequential stages are data preparation, base evaluation, QLoRA training, adapter evaluation, and paired report. First model download and container build require network access to Docker Hub, PyPI, and Hugging Face. The image disables Hugging Face's Xet transfer path because the pinned BANKING77 download returned a CDN 403 through this lab connection; ordinary HTTPS succeeded.

When the Job is `Complete`:

```sh
bash scripts/open-results.sh
cat runs/report.md
```

This starts a CPU results pod, copies the small reports into the Dev Spaces workspace, and serves TensorBoard from the persisted training logs. To view the curves from a laptop on the VPN, run `oc -n YOUR_GPU_PROJECT port-forward pod/gemma4-results 6006:6006` on that laptop, then open `http://localhost:6006`. Delete the reader pod when done: `oc -n YOUR_GPU_PROJECT delete pod gemma4-results`.

Dev Spaces keeps source code on its configured **10 GiB `local-path` workspace PVC**. The much larger model cache, adapter, predictions, and logs use a separate **80 GiB Ceph RBD PVC**. The workspaces' node-local storage is unsuitable as the model cache for this experiment.

## Reproducibility and boundaries

- Model, tokenizer, and dataset are pinned to Hugging Face revisions in [common.py](banking_lab/common.py); Python dependencies and the PyTorch image have pinned versions.
- BANKING77 is [CC BY 4.0](https://huggingface.co/datasets/PolyAI/banking77); attribute PolyAI and the original dataset authors if you redistribute derived material. The Gemma 4 model card lists Apache 2.0.
- The default 385-row balanced subset is for instruction and rapid comparison, **not** the official full-test benchmark. Do not choose hyperparameters by looking at held-out test results; use validation loss, then run the test comparison once.
- This is intent classification, not banking advice, transaction authorization, or a production service. A small discriminative classifier may be a better production baseline. The lab demonstrates SFT mechanics rather than proving that fine tuning is necessary.
- No Hugging Face or OpenShift token is needed in this repository. The public model/dataset download without a token; Dev Spaces uses your OpenShift login. Never commit kubeconfigs, tokens, or private customer messages. Generated `runs/` and environment files are ignored by Git.
- The code and manifests are validated locally, but **GPU training and end-to-end Dev Spaces execution from this new repository have not yet been run**.

## Repository map

| Path | Purpose |
|---|---|
| `banking_lab/prepare.py` | Pinned dataset download and deterministic splits |
| `banking_lab/train.py` | TRL SFT and PEFT QLoRA adapter |
| `banking_lab/evaluate.py` | Base or adapter generation and scoring |
| `banking_lab/report.py` | Paired comparison |
| `deploy/` | 80 GiB PVC, H200 Job, optional TensorBoard reader pod |
| `scripts/` | OpenShift build/run and results commands |
| `tests/` | Split-leakage and scoring checks |
