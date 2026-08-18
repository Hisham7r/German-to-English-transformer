# German-to-English Neural Machine Translation (Transformer from scratch)

## Overview

From-scratch PyTorch implementation of the Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017, NeurIPS), translating German to English. Not a wrapper around a pretrained model or HuggingFace pipeline — embeddings, positional encoding, multi-head attention, encoder, decoder are being built and trained from first principles to develop deep mechanical understanding, not just usage fluency.

**Purpose:** Portfolio-building and skill development for a transition into an AI Engineer role. Background: software/web development, strong classical ML theory (regression, trees, ensembles, SVMs, basic neural nets), limited hands-on deep learning experience.

**Reference implementations for guidance (not copied):** The Annotated Transformer (Harvard NLP), Karpathy's minGPT, fairseq.

## Dataset

**Multi30k** — ~31k German/English image caption pairs (images unused, only parallel text). Short, clean sentences (~12 words avg). Loaded via `datasets.load_dataset("bentrevett/multi30k")` — gives `train`/`validation`/`test` splits, each with `en`/`de` fields.

## Hardware / Workflow

- Local dev machine: Intel i5 (6th gen), 4GB RAM, 256GB SSD, **no GPU**.
- Code is written and developed locally (repo structure, tokenizer, model code, training loop logic).
- Actual training runs happen on **Google Colab** (free-tier GPU, e.g. T4). Code pushed to GitHub, pulled into Colab notebooks for training, checkpoints/results pulled back locally.
- Implication: keep training-loop code Colab-portable (no local-only paths/assumptions), and keep local-only work to things that don't need a GPU (architecture code, data pipeline, sanity checks on tiny subsets).

## Project Structure (target)

```
translator/
├── configs/
│   └── config.yaml          # all hyperparameters, paths, settings — nothing hardcoded in code
├── data/                     # raw/processed data (gitignored)
├── src/
│   ├── data/                  # tokenizer, vocabulary, Dataset/DataLoader code
│   ├── models/                # Transformer architecture components
│   ├── train.py
│   └── evaluate.py
├── experiments/               # logs, checkpoints, run outputs (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

Core dependencies: `torch`, `datasets`, `sacrebleu` (BLEU evaluation).

Config-driven from day one: nothing hardcoded in code if it belongs in `config.yaml`.

## Project Phases

1. **Reproduction** — build and train the Transformer from scratch on Multi30k, reach a working, evaluated baseline with a real BLEU score.
2. **Improvements** — controlled experiments beyond baseline: BPE vs. word-level tokenization, beam search vs. greedy decoding, LR schedule tuning, knowledge distillation, LoRA-style adaptation. Each improvement = a stated hypothesis, measured before/after against baseline.
3. **Production engineering** — config management, logging, experiment tracking, testing, reproducibility (seeding), FastAPI serving endpoint (German in, English out), Docker, documentation.
4. **Open source contribution** (not part of this project directly, but the prep goal) — eventually contribute to repos like sentence-transformers, lm-evaluation-harness, huggingface/transformers.

## Detailed Build Flow

**Week 1 — Data pipeline and baseline**
- Load Multi30k, inspect directly (print examples, check anomalies/empty strings/encoding issues)
- Decide tokenization strategy (word-level vs. subword/BPE) and vocabulary construction (frequency cutoff, OOV handling)
- Decide special tokens needed (SOS/EOS/PAD/UNK) and why a seq2seq model needs them
- Custom PyTorch `Dataset`/`DataLoader` with a `collate_fn` that pads batches and produces attention masks
- No-model baseline (e.g. copy source unchanged, or word-for-word dictionary lookup) + its BLEU score as the number every future result must beat

**Week 2 — Model skeleton**
- Token embeddings, positional encoding, scaled dot-product attention, multi-head attention, position-wise feed-forward, encoder stack, decoder stack, output projection
- Sanity check: deliberately overfit a tiny subset (~10 examples) to confirm model/loss wiring before scaling up

**Week 3 — Training loop**
- Hand-written loop, no high-level Trainer abstractions
- LR warmup schedule per the paper, label smoothing
- Full training run on Multi30k (on Colab GPU)

**Week 4 — Evaluation and decoding**
- Greedy (autoregressive) decoding
- BLEU on held-out test set via `sacrebleu`
- Compare against Week 1 no-model baseline
- Qualitative error analysis on actual outputs vs. references

**Weeks 5–6 — Improvements (Phase 2)**
- Beam search instead of greedy
- BPE instead of word-level (if word-level was the initial choice)
- Possible: knowledge distillation, LoRA-style adaptation
- Each change = controlled experiment, stated hypothesis, before/after BLEU

**Weeks 7–8 — Production layer (Phase 3)**
- Experiment tracking (MLflow or W&B)
- Full config-driven setup
- FastAPI endpoint (German → English)
- Dockerize serving
- Polish docs/README for portfolio presentation

## Pending Decisions (deliberately being worked through as learning exercises — don't just hand over the answer)

1. Word-level vs. subword (BPE) tokenization, and the reasoning given dataset size (~29k pairs).
2. Vocabulary frequency cutoff strategy and OOV handling at inference time.
3. Which special tokens a seq2seq translation model needs (vs. a classifier), and why each is needed.

## Working style for this project

The user wants to *understand* the Transformer architecture and training pipeline deeply, not just get working code. For open design decisions (tokenization strategy, vocab cutoffs, special tokens, architecture choices), prefer walking through the reasoning and trade-offs collaboratively rather than immediately prescribing the answer — these are explicitly learning exercises. Once a decision is made, implementation can proceed normally.

## Portfolio Goal

End deliverable: a working, evaluated, documented translation system in a public GitHub repo demonstrating research paper comprehension + implementation, from-scratch deep learning architecture work, proper experimental methodology/evaluation, and production-quality engineering (config, logging, serving, containerization).

## Progress tracking

See [PROGRESS.md](PROGRESS.md) for the architecture walkthrough (end-to-end data/request flow, repo structure explained), the reasoning behind decisions already made, and a dated build log. It's meant to let a newcomer (human or agent) understand the whole project from scratch without needing prior conversation history.

**Update PROGRESS.md whenever a meaningful change is made** — a new component, a decision finalized, a structural change. Add a dated entry to the Progress Log, and update the architecture/decisions sections if they're now stale. Don't log every trivial edit — log steps that change how the system works or what exists.
