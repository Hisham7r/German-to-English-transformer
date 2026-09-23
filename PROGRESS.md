# Project Progress

**Single source of truth for implementation progress.** A new agent or developer should be able to read this file and know what exists, what works, what is being built right now, and what to do next. Goals, constraints and agent working rules live in [CLAUDE.md](CLAUDE.md); this file tracks the *build*. Maintenance rules are in CLAUDE.md → "Project Progress Tracking".

---

## 1. Project Overview

A German→English translation system built by implementing the Transformer ("Attention Is All You Need", Vaswani et al., 2017) **from scratch in PyTorch** — no pretrained models, no HuggingFace `pipeline()`. Trained on Multi30k (29,000 train / 1,014 validation / 1,000 test sentence pairs).

- **Purpose:** portfolio piece and deep-learning skill-building for a move into an AI Engineer role. Understanding the mechanics matters more than the BLEU score.
- **Target end state:** a documented, evaluated translator with a FastAPI endpoint (German in, English out), containerized with Docker.
- **Hardware split:** code is written locally (i5, 4 GB RAM, **no GPU**); real training runs on Google Colab (free-tier GPU). Local runs are limited to tiny sanity checks.
- **Development stage:** Phase 1 (Reproduction), **Week 3 of 8 — training loop, early stage.** The full model exists and is verified; no real training run has happened yet.

Pipeline at a glance:

```
Multi30k ─► BPE tokenizer (shared, 8000) ─► Dataset/DataLoader ─► Transformer (6+6 layers, d_model 512)
                                                                        │
   [built, verified] ◄──────────────────────────────────────────────────┘
   [NOT BUILT YET]   training loop (Adam + warmup LR + label smoothing) ─► greedy decode ─► BLEU vs 0.48 baseline ─► FastAPI/Docker
```

Repository map (tracked files):

```
CLAUDE.md, PROGRESS.md, README.md (2-line stub), requirements.txt, .gitignore, .vscode/settings.json
configs/config.yaml            all hyperparameters: dataset, tokenizer, model (512/6), sanity_check (128/2)
data/bpe_tokenizer.json        trained tokenizer (tracked on purpose; combined_corpus.txt is gitignored)
src/explore.py                 dataset inspection
src/baseline.py                no-model BLEU baseline
src/data/tokenizer.py          builds corpus + trains shared BPE tokenizer
src/data/dataset.py            TranslationDataset + collate_fn
src/models/                    embeddings, attention, multi_head_attention, feed_forward, add_norm,
                               encoder, decoder, transformer, overfit_sanity_check
src/training/                  lr_schedule.py, label_smoothing.py   (standalone pieces, not yet wired in)
```

Not yet created: `src/training/train.py` (or `src/train.py`), `src/evaluate.py`, any `experiments/` output.

---

## 2. Current Status

| Area | Status | Notes |
|---|---|---|
| Week 1 — data pipeline + baseline | ✅ Completed | Baseline BLEU **0.48** |
| Week 2 — model architecture | ✅ Completed | Full Transformer, weight-tied |
| Week 2 — overfit sanity check | ✅ Completed | Loss 78.6 → ~0 on 10 pairs (small config) |
| Week 3 — LR warmup schedule | ✅ Completed (standalone) | `src/training/lr_schedule.py`; not wired to an optimizer |
| Week 3 — label smoothing | 🟡 In Progress | Only a toy comparison demo exists; not in a real loss/criterion |
| Week 3 — training config (`training:` in config.yaml) | 🔴 Not Started | Batch size, warmup steps, Adam settings undecided |
| Week 3 — training loop (`train.py`) | 🔴 Not Started | |
| Week 3 — full Colab training run | 🔴 Not Started | |
| Week 4 — greedy decoding + BLEU eval | 🔴 Not Started | |
| Weeks 5–6 — improvements (beam search, etc.) | 🔴 Not Started | |
| Weeks 7–8 — serving / Docker / tracking | 🔴 Not Started | |
| AddNorm dropout placement vs. paper | ⚠️ Needs Verification | See Known Issues #1 |
| Full-size model training behavior (memory, speed, convergence) | ⚠️ Needs Verification | Only forward-pass tested at full size |

---

## 3. Completed Work

### Week 1 — Data pipeline and baseline (2026-08-19 → 2026-08-22)
- **Setup:** repo scaffold, `.venv`, `requirements.txt`, config-driven design (`configs/config.yaml`).
- **Data inspection** (`src/explore.py`): split sizes confirmed; sample pairs clean, German umlauts/ß intact.
- **Tokenizer** (`src/data/tokenizer.py`): one shared BPE tokenizer for both languages, vocab 8000, special tokens `<pad>=0 <sos>=1 <eos>=2 <unk>=3`, `Whitespace` pre-tokenizer, trained on the **train split only** (no val/test leakage). Output: `data/bpe_tokenizer.json`.
- **Dataset/DataLoader** (`src/data/dataset.py`): tokenizes all pairs once up front, wraps with `<sos>/<eos>`, and splits each target into `decoder_input`/`decoder_target` *before* padding. `collate_fn` pads per-batch and returns padding masks.
- **Baseline** (`src/baseline.py`): copy German unchanged, score with `sacrebleu` on the 1,000-sentence test set → **BLEU 0.48**. This is the floor every trained result must beat.

### Week 2 — Model architecture (2026-08-24 → 2026-09-13)
All in `src/models/`, each file runnable standalone with a shape/sanity test block:
- `embeddings.py` — `TokenEmbedding` (scaled by √d_model), `PositionalEncoding` (sin/cos + dropout).
- `attention.py`, `multi_head_attention.py` — scaled dot-product and 8-head attention; supports different query/key lengths (needed for cross-attention) and masking.
- `feed_forward.py`, `add_norm.py` — position-wise FFN (d_ff 2048); residual + LayerNorm + dropout.
- `encoder.py`, `decoder.py` — 6-layer stacks; decoder has masked self-attention + cross-attention; `generate_causal_mask`.
- `transformer.py` — wires encoder + decoder + output projection; **weight tying** across encoder embedding, decoder embedding and output projection; builds source padding mask and target (causal × padding) mask internally.
- `overfit_sanity_check.py` — trains a **small** model (d_model 128, 2 layers, 1.7M params) on 10 real pairs for 500 steps. Passed on 2026-09-13: loss 78.6 → ~0, near-perfect reproduction of all 10 sentences.

---

## 4. Current Phase

**Phase 1 (Reproduction) → Week 3: Training loop.** Started 2026-09-19.

- **Done in this phase:**
  - `src/training/lr_schedule.py` — `get_warmup_lr_lambda(d_model, warmup_steps)` implements the paper's Section 5.3 schedule. Output matches the reference values (step 1 ≈ 1.75e-7, peak ≈ 6.99e-4 at step 4000).
  - `src/training/label_smoothing.py` — toy demo (ε = 0.1) showing smoothing penalizes overconfidence (gap 0.54 when confident vs 0.10 when unsure). Educational only.
- **Remaining in this phase:** training config, real criterion with label smoothing, optimizer wiring, `train.py` with validation loop + checkpointing, local smoke test, full Colab run.
- **Known blockers:** none. Open design decisions (batch size, warmup steps, epochs) need discussion before coding — see Next Steps.

> Note: commit messages `14c8220` ("the warmup stage") and `732779e` ("added label smoothing to transformer") overstate what changed — both only add the standalone scripts above. The model and loss code were not modified.

---

## 5. Next Steps

1. **Define the `training:` section in `config.yaml`** (collaboratively — these are open decisions): batch size, epochs, `warmup_steps`, label smoothing ε, Adam β/ε, gradient-clip norm, seed. The paper's `warmup_steps=4000` was tuned for WMT-scale data; with 29k pairs, batch 64 gives ~450 steps/epoch, so 4000 warmup steps ≈ 9 epochs — likely too long, so decide deliberately.
2. **Resolve the AddNorm dropout question** (Known Issues #1) *before* the full run, since changing architecture afterwards wastes Colab time.
3. **Write the real criterion:** `nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=ε)`.
4. **Write `train.py`:** DataLoader from `dataset.py`, Adam with base `lr=1.0` + `LambdaLR(get_warmup_lr_lambda(...))` (the lambda returns the *absolute* LR, so the base LR must be 1.0), gradient clipping, per-epoch validation loss, checkpoints to `experiments/`, seeding.
5. **Smoke-test `train.py` locally** with the small `sanity_check` config on a tiny subset.
6. **Run the full-size model on Colab** (tokenizer file is tracked in git, so it is available after `git clone`).
7. **Week 4:** greedy decoding, a decode helper that rejoins BPE pieces, BLEU on the test set vs the 0.48 baseline, error analysis.

---

## 6. Important Technical Changes

| Date | Change | Why it matters |
|---|---|---|
| 2026-08-22 | `.gitignore`: `data/` → `/data/` | The unanchored rule silently excluded `src/data/` from git; `dataset.py`/`tokenizer.py` were untracked until fixed. |
| 2026-08-22 | Decoder input/target shift moved into `Dataset.__getitem__` (pre-padding) | Teacher forcing needs `[sos,…]` in / `[…,eos]` out; shifting raw sequences is unambiguous. |
| 2026-08-28 → 09-13 | All notebooks converted to `.py` | Clean diffs, importable modules, terminal-based workflow. |
| 2026-09-13 | Weight tying added in `Transformer` | One shared 8000×512 matrix for encoder/decoder embeddings and output projection. |
| 2026-09-13 | Added `sanity_check:` config (128/2) | Full-size model is too slow to overfit on local CPU; real `model:` (512/6) untouched for Colab. |
| 2026-09-19 | `data/bpe_tokenizer.json` now tracked (`/data/*` + `!/data/bpe_tokenizer.json`) | Needed on Colab without re-training; `combined_corpus.txt` remains ignored. |
| 2026-09-19 | Added `src/training/` package | Home for schedule, loss and (later) the training loop. |

No new third-party dependencies since `requirements.txt` was created (torch, datasets, sacrebleu, pyyaml, tokenizers).

---

## 7. Known Issues / Technical Debt

| # | Problem | Location | State | Blocks work? |
|---|---|---|---|---|
| 1 | ⚠️ **AddNorm applies dropout *after* LayerNorm** (`dropout(norm(x + sublayer))`). The paper (Sec. 5.4) applies dropout to the sub-layer output *before* the residual add and normalization. Needs a decision; it is a locked "Post-LN" design, so this is a placement detail, not Pre-LN. | `src/models/add_norm.py` | Unresolved | No, but decide before the full training run |
| 2 | Decoding leaves BPE pieces unjoined (`play house`, `li on`) — the tokenizer has no decoder set. Also, positions after `<eos>` output repeated junk (expected: loss ignores padding there). | `src/data/tokenizer.py`; visible in `overfit_sanity_check.py` output | Unresolved | Not Week 3; **must fix for Week 4 BLEU** |
| 3 | Source-side conventions differ: `dataset.py` wraps source with `<sos>/<eos>`, the sanity-check script does not. `collate_fn` also returns masks that `Transformer` ignores (it derives its own from `pad_id`). | `src/data/dataset.py`, `src/models/overfit_sanity_check.py`, `src/models/transformer.py` | Unresolved | No; settle when writing `train.py` |
| 4 | No `training:` config section; once `train.py` exists, hyperparameters must not be hardcoded there. | `configs/config.yaml` | Unresolved | Yes for step 1 of Next Steps |
| 5 | `lr_schedule.py` has inconsistent 8-space indentation and no imports (unused, harmless). Cosmetic. | `src/training/lr_schedule.py` | Runs correctly | No |
| 6 | `README.md` is a 2-line stub. | `README.md` | Deferred to Phase 3 polish | No |

Environment gotchas (not bugs):
- The Bash tool's terminal mangles UTF-8 (umlauts show as `�`); use PowerShell to view German text. Data on disk is correct.
- Scripts that import `datasets` print a harmless `ResourceTracker … _recursion_count` traceback at exit on Windows.
- `HF_TOKEN` warning and occasional `getaddrinfo failed` retries are harmless once the dataset is cached.

---

## 8. Testing / Verification Status

Every script is self-testing; run from the project root with the venv active (`python <path>`). "Re-verified" means re-run on 2026-09-24.

| Component | Status | Evidence |
|---|---|---|
| `explore.py` | ✅ Tested and working | Verified at notebook→script conversion (2026-08-28); not re-run |
| `data/tokenizer.py` | ✅ Tested and working | Vocab 8000, sensible splits (2026-09-13). Not re-run: it overwrites the tracked tokenizer file |
| `data/dataset.py` | ✅ Re-verified | Batch shapes + decoder_input/target shift correct |
| `baseline.py` | ✅ Re-verified | BLEU 0.48 on 1,000 test sentences |
| `embeddings`, `attention`, `multi_head_attention`, `feed_forward`, `add_norm` | ✅ Re-verified | Output shapes correct; masked positions get weight 0.0 and rows sum to 1 |
| `encoder`, `decoder` | ✅ Re-verified | `[2,10]→[2,10,512]`; decoder handles src/tgt length mismatch; causal mask correct |
| `transformer.py` | ✅ Re-verified | logits `[2,8,8000]` at **full size** (512/6); weight tying `is` checks True |
| `overfit_sanity_check.py` | ✅ Tested and working | Passed 2026-09-13; not re-run (~1 min; file unchanged since) |
| `training/lr_schedule.py` | ✅ Re-verified | Matches reference values at steps 1, 2000, 4000, 16000 |
| `training/label_smoothing.py` | ✅ Re-verified | Demo output matches expectation (gap 0.54 vs 0.10) |
| Full-size training (speed, memory, convergence) | ⚠️ Not verified | Never run beyond a forward pass; a full-size 500-step CPU run was abandoned after 15+ min |
| Real translation quality | 🔴 Not tested | No trained model yet |

There is no automated test suite (pytest) yet; testing is via the `__main__` blocks above.

---

## 9. Important Decisions / References

Locked decisions (also listed in CLAUDE.md — do not revisit unless the user asks) and their reasoning:

- **BPE, not word-level.** German compounds (e.g. `Antriebsradsystem`) would be rare/OOV as words; BPE splits them into reusable pieces. Matches the paper.
- **Shared vocabulary, one tokenizer.** Matches the paper's En–De setup and is what makes weight tying possible.
- **Vocab 8000, not ~37k.** The paper's size suits WMT-scale data; with 29k pairs a larger vocab leaves most tokens too rare to learn good embeddings.
- **Special tokens `<pad> <sos> <eos> <unk>`.** `<sos>/<eos>` let the decoder start and know when to stop; `<pad>` enables batching; `<unk>` is the fallback.
- **Tokenizer trained on train split only** — keeps evaluation honest.
- **Decoder shift before padding, on raw sequences** — unambiguous and standard.
- **Weight tying** — fewer parameters, one consistent token representation.
- **Post-LN as in the paper** (LayerNorm after the residual add). See Known Issues #1 for the dropout-placement detail.
- **Two model configs:** `model:` 512/6 for Colab; `sanity_check:` 128/2 for local CPU verification only.
- **All `.py`, no notebooks; no `Co-Authored-By` lines in commits** (see CLAUDE.md).

References: [CLAUDE.md](CLAUDE.md) (goals, hardware, agent rules, build plan by week), `configs/config.yaml` (all hyperparameters), the paper's Sections 3–5 (architecture, Sec. 5.3 LR schedule, Sec. 5.4 regularization).

---

Last Updated: 2026-09-24
