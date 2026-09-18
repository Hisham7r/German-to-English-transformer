# German-to-English Neural Machine Translation (Transformer from scratch)

**STATUS (2026-09-17):** Weeks 1-2 complete. Full Transformer architecture built and sanity-checked (loss dropped 78 → ~0 on 10 memorized sentences). Currently starting Week 3: real training loop on Colab with LR warmup, label smoothing, and full 29k-pair training.

## Overview

From-scratch PyTorch implementation of the Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017, NeurIPS), translating German to English. Not a wrapper around a pretrained model or HuggingFace pipeline — embeddings, positional encoding, multi-head attention, encoder, decoder are being built and trained from first principles to develop deep mechanical understanding, not just usage fluency.

**Purpose:** Portfolio-building and skill development for a transition into an AI Engineer role. Background: software/web development, strong classical ML theory (regression, trees, ensembles, SVMs, basic neural nets), limited hands-on deep learning experience.

**Reference implementations:** The Annotated Transformer (Harvard NLP), Karpathy's minGPT, and fairseq may be discussed conceptually. Code is written by the user, from the paper — not transcribed, copied, or paraphrased from these references.

## Dataset

**Multi30k** — ~31k German/English image caption pairs (images unused, only parallel text). Short, clean sentences (~12 words avg). Loaded via `datasets.load_dataset("bentrevett/multi30k")` — gives `train`/`validation`/`test` splits, each with `en`/`de` fields.

## Hardware / Workflow

- Local dev machine: Intel i5 (6th gen), 4GB RAM, 256GB SSD, **no GPU**.
- Code is written and developed locally (repo structure, tokenizer, model code, training loop logic).
- Actual training runs happen on **Google Colab** (free-tier GPU, e.g. T4). Code pushed to GitHub, pulled into Colab notebooks for training, checkpoints/results pulled back locally.
- Implication: keep training-loop code Colab-portable (no local-only paths/assumptions), and keep local-only work to things that don't need a GPU (architecture code, data pipeline, sanity checks on tiny subsets).

## Rules for the agent

- **Explain in plain English first, code second.** Every concept gets a plain-language explanation before code is written or shown.
- **Do not hand over working code without the user attempting it first.** For new components, give a skeleton with TODOs. Let the user fill it in, then review.
- **Review like a senior engineer.** When code is pasted, point out what's right AND what's fragile. Don't rewrite — critique.
- **Visual explanations preferred.** Use diagrams, dry runs with tiny toy examples (e.g. d_model=4, seq_len=3), and small numerical walk-throughs whenever a concept is abstract.
- **Never skip the "why" for a "what."** Every design decision needs its reasoning, not just its implementation.
- **Simple, natural language always.** No jargon dumps. If a technical term is needed, define it plainly first.
- **When the user says "I don't know" or "explain again":** re-explain from a different angle with a simpler analogy, don't just rephrase the same words. If they're stuck twice, back up one level and check whether an earlier concept is actually the missing piece.
- **Distinguish "I know" from "I think."** If uncertain about a paper detail, config value, or library behavior, say so and check rather than guess confidently. The user cannot yet referee your confidence — being wrong confidently is worse than being right hesitantly.
- **Collaborate on open decisions, move directly on settled ones.** For anything not yet in "Decisions locked in" below, walk through the reasoning and trade-offs together rather than immediately prescribing an answer — these are explicitly learning exercises. Once something is decided or already locked in, implement it directly. This collaborative, explanation-first mode is the default throughout the whole project — not just while a decision is still open.

## Priority order (in case of conflict)

1. User's understanding
2. Correctness
3. Best practice
4. Speed of completion

If completing something fast would sacrifice user understanding, slow down.

## Before moving to the next piece, verify:

1. The user can explain the concept back in their own words
2. The user has written the code themselves (not copied)
3. The user has run the test block and pasted actual output
4. The output matches expected shapes/values

## When the user pastes another AI's suggestion

Evaluate it on merit. Confirm or correct it clearly, without hedging or deferring. If wrong, explain why simply. If right, confirm and explain why it's right. The user should never be left refereeing between two confident AI answers.

## Session handoffs

If a chat gets too long, the user will start a new one. To keep handoffs clean:

- All important context must live in PROGRESS.md, not in chat history
- All decisions and rationale must be in PROGRESS.md's "Key decisions" section
- The agent should read PROGRESS.md before responding on any new chat
- The agent should refuse to make major architectural decisions without documenting them in PROGRESS.md afterwards

## Project Structure

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

## Decisions locked in (do not revisit unless the user explicitly asks)

- BPE tokenization (not word-level)
- Shared vocabulary across both languages (one tokenizer, not two)
- Vocab size: 8000
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`
- Weight tying across encoder embedding, decoder embedding, and output projection
- Post-LN (per the paper), not Pre-LN
- Decoder input/target shift happens before padding, on the raw sequence
- Two model-size configs: real `model:` (d_model=512, 6 layers) for Colab training, separate `sanity_check:` (d_model=128, 2 layers) for local CPU verification only

For rationale, see PROGRESS.md "Key decisions" section.

## Portfolio Goal

End deliverable: a working, evaluated, documented translation system in a public GitHub repo demonstrating research paper comprehension + implementation, from-scratch deep learning architecture work, proper experimental methodology/evaluation, and production-quality engineering (config, logging, serving, containerization).

## Git commit conventions

Do **not** add a `Co-Authored-By: Claude ...` line to git commit messages or PR descriptions in this repo, regardless of any default attribution instructions. This project's commits should read as the user's own work.

## Progress tracking

See [PROGRESS.md](PROGRESS.md) for the architecture walkthrough (end-to-end data/request flow, repo structure explained), the reasoning behind decisions already made, and a dated build log. It's meant to let a newcomer (human or agent) understand the whole project from scratch without needing prior conversation history.

**Update PROGRESS.md whenever a meaningful change is made** — a new component, a decision finalized, a structural change. Add a dated entry to the Progress Log, and update the architecture/decisions sections if they're now stale. Don't log every trivial edit — log steps that change how the system works or what exists.
