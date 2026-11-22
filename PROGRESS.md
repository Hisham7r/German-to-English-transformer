  ## START HERE

  - **Project:** a German to English Transformer written from scratch in PyTorch.
  - **Official result:** greedy decoding on all 1,000 test sentences, with `<sos>/<eos>` German input (same as training): **BLEU 37.58**. Beam 4 on the test set: **not run**.
  - **Rule:** tune on the validation split only. The test split is for final numbers.
  - **Beam 4 is on hold:** +0.69 BLEU on 200 validation sentences, **not proven** (bootstrap: beam 4 higher in 90.4% of resamples, range -0.29 to 2.41), and about 19x slower than greedy (9.63 s vs 0.52 s per sentence).
  - **Next step:** Phase 3 (serving, Docker, tracking). Training stops (best checkpoint epoch 33, val loss 1.7422). The glue-rule check is done and the old rule is kept. A tokenizer retrain is deferred as an optional later experiment; an alpha test is not decided and not run.
  - **Backup:** `best_model_weights.pt` is backed up (confirmed by the user). It is not in git. `best_model.pt` (552.8 MiB, with optimizer state, only needed to resume training): backup **not confirmed**.
  - **Check scripts** (`src/check_*.py`):
    - `check_beam.py`: beam size 1 against greedy (identical ids?) and a beam-size table on the first 200 test sentences (run before the input fix).
    - `check_src_markers.py`: greedy on the first 200 validation sentences, German input without vs. with `<sos>/<eos>`.
    - `check_beam_gain.py`: checks the joined beam 4 validation file, then a paired bootstrap of beam 4 vs. beam 1.
    - `check_unseen_words.py`: share of output words that never appear in the training English (first 200 validation sentences).
    - `check_text_fixes.py`: BLEU effect of re-gluing spaced hyphens and apostrophes in the output (first 200 validation sentences).
    - `check_glue_rule.py`: greedy on the first 200 validation sentences, old single-id word-gluing rule vs. a previous-id rule (not yet committed).

  ---

  # Project Progress

  **Single source of truth for implementation progress.** A new agent or developer should be able to read this file and know what exists, what works, what is being built right now, and what to do next. Goals, constraints and agent working rules live in [CLAUDE.md](CLAUDE.md); this file tracks the *build*. Maintenance rules are in CLAUDE.md → "Project Progress Tracking".

  > **Warning:** Earlier BLEU and repetition numbers (everything except the 0.48 baseline) were measured with German input WITHOUT `<sos>/<eos>`, which differs from training. That bug is fixed (Known Issues #11). The old numbers are kept in one place only, the note "Measured with the wrong input format" in Section 4; all other numbers in this file use the correct input. Anything not listed as measured is **unmeasured**, notably beam 4 on the full test set with markers.

  ---

  ## 1. Project Overview

  A German→English translation system built by implementing the Transformer ("Attention Is All You Need", Vaswani et al., 2017) **from scratch in PyTorch** — no pretrained models, no HuggingFace `pipeline()`. Trained on Multi30k (29,000 train / 1,014 validation / 1,000 test sentence pairs).

  - **Purpose:** portfolio piece and deep-learning skill-building for a move into an AI Engineer role. Understanding the mechanics matters more than the BLEU score.
  - **Target end state:** a documented, evaluated translator with a FastAPI endpoint (German in, English out), containerized with Docker.
  - **Hardware split:** code is written locally (i5, 4 GB RAM, **no GPU**); real training runs on Google Colab (free-tier GPU). Local runs are limited to tiny sanity checks.
  - **Development stage:** Phase 1 (Reproduction), **Weeks 3 and 4 complete; beam search (first Phase 2 experiment) measured on 200 validation sentences only, gain over greedy not proven.** A real full-size training run on Colab finished: best validation loss **1.74** (perplexity **5.71**) at epoch 33 of a configured 40. Test set (1,000 sentences, German input with `<sos>/<eos>`, matching training), BLEU: no-model baseline **0.48**, greedy **37.58**; beam 4 on the full test set with markers is **unmeasured**. Remaining: more error analysis and the Phase 2 follow-ups (see Next Steps).

  Pipeline at a glance:

  ```
  Multi30k ─► BPE tokenizer (shared, 8000) ─► Dataset/DataLoader ─► Transformer (6+6 layers, d_model 512)
                                                                          │
    [built, verified] ◄──────────────────────────────────────────────────┘
    training loop (Adam + warmup LR + label smoothing) ─► RUN ON COLAB, val loss 1.74 ─► experiments/best_model.pt
                                                                          │
    [built, verified] ◄──────────────────────────────────────────────────┘
    greedy / beam decode (decode.py) ─► clean_decode rejoin (detokenize.py) ─► BLEU 0.48 baseline / 37.58 greedy on test, with markers (evaluate.py)
                                                                          │
    [built, verified] ◄──────────────────────────────────────────────────┘
    error analysis (analyze_outputs.py) ─► FastAPI/Docker
  ```

  Repository map (tracked files):

  ```
  CLAUDE.md, PROGRESS.md, README.md (2-line stub), requirements.txt, .gitignore, .vscode/settings.json
  configs/config.yaml            all hyperparameters: dataset, tokenizer, model (512/6), sanity_check (128/2)
  data/bpe_tokenizer.json        trained tokenizer (tracked on purpose; combined_corpus.txt is gitignored)
  src/explore.py                 dataset inspection
  src/baseline.py                no-model BLEU baseline
  src/data/tokenizer.py          builds corpus + trains shared BPE tokenizer
  src/data/dataset.py            TranslationDataset + collate_fn + encode_source (the one German-input encoder, used by training and decoding)
  src/models/                    embeddings, attention, multi_head_attention, feed_forward, add_norm,
                                encoder, decoder, transformer, overfit_sanity_check
  src/training/                  lr_schedule.py, label_smoothing.py   (standalone pieces)
  src/train.py                   training loop (train/val, warmup LR, label smoothing, checkpointing); `--smoke` = tiny local run
  src/decode.py                  greedy_decode() + beam_search_decode() + a single-sentence CLI check against experiments/best_model.pt
  src/data/detokenize.py         heuristic BPE rejoin (build_boundary_map/clean_decode) -- see Known Issues #2
  src/evaluate.py                BLEU on the test or validation split (beam search; --beam-size/--alpha/--split/--start/--limit); results file names carry a _markers suffix (--split committed in a6a1dfa)
  src/analyze_outputs.py         quantifies repetition + punctuation-spacing effect on experiments/test_translations.txt
  src/check_beam.py              beam-vs-greedy correctness check + beam-size table on the first 200 test sentences (pushed 2026-10-08)
  src/check_src_markers.py       greedy A/B on the first 200 validation sentences: German input without vs. with markers
  (not yet committed: src/score_results.py -- scores saved HYP/REF files; src/check_beam_gain.py -- join check + paired bootstrap, beam 1 vs. beam 4)
  ```

  `experiments/` holds `best_model.pt` (580MB, the real Colab run, with optimizer/scheduler state), `best_model_weights.pt` (194MB, same run, only `epoch`/`best_val_loss`/`model_state` -- no optimizer state, created 2026-10-08; presumably for easier transfer to Colab, since nothing in the repo loads it yet), result files (`test_translations*.txt`, `validation_translations*.txt`) and local smoke-test checkpoints; all gitignored.

  ---

  ## 2. Current Status

  | Area | Status | Notes |
  |---|---|---|
  | Week 1 — data pipeline + baseline | ✅ Completed | Baseline BLEU **0.48** |
  | Week 2 — model architecture | ✅ Completed | Full Transformer, weight-tied |
  | Week 2 — overfit sanity check | ✅ Completed | 10 pairs, small config; re-run 2026-09-24 after the init fix: loss 9.6 → 0.003 |
  | Week 3 — LR warmup schedule | ✅ Completed | `src/training/lr_schedule.py`; wired into `train.py` via `LambdaLR` |
  | Week 3 — label smoothing | ✅ Completed | Used in `train.py`'s training criterion (ε from config); validation deliberately unsmoothed |
  | Week 3 — training config (`training:` in config.yaml) | ✅ Completed | batch 128, **40 epochs, warmup 4000** (retuned after divergence — see Technical Changes), smoothing 0.1 |
  | Week 3 — training loop (`src/train.py`) | ✅ Completed | Smoke-tested locally, then run for real on Colab |
  | Embedding init scale (tied embeddings) | ✅ Completed | Fixed 2026-09-24: initial loss at full size 344.6 → 12.9 (uninformed model ≈ 9.0) |
  | Week 3 — full Colab training run | ✅ Completed | Best checkpoint epoch 33/40: **val loss 1.74, perplexity 5.71** (`experiments/best_model.pt`, 580MB, not in git) |
  | Week 4 — greedy decoding (`src/decode.py`) | ✅ Completed | Verified on 3 real sentences; correctly reaches `<eos>`; repetition is a known greedy-decoding limitation (#7), not a bug |
  | Week 4 — BPE-rejoin heuristic (`src/data/detokenize.py`) | ✅ Completed | Frequency-based word-boundary heuristic (Known Issues #2); fixes split words; stray space before punctuation confirmed present but shown **not to affect BLEU** (`sacrebleu` normalizes it away internally) |
  | Week 4 — BLEU eval on test set (`src/evaluate.py`) | ✅ Completed | Full 1,000-sentence test set against `experiments/best_model.pt`, German input with markers: baseline **0.48**, greedy **37.58** (67.7/44.9/30.6/21.4). Old no-marker scores: see the note in Section 4. |
  | Week 4 — quantitative error analysis (`src/analyze_outputs.py`) | ✅ Completed | The repetition analysis was run on no-marker output, so its percentages live only in the Section 4 note. With markers, test greedy has hard loops 3.0% and repeated 3-word phrase 5.1% (`evaluate.py` definitions). See Known Issues #7. |
  | Weeks 5–6 — beam search (`beam_search_decode` in `src/decode.py`, `decoding:` config, `evaluate.py --beam-size/--alpha`) | ✅ Implemented and verified; measured on 200 validation sentences only | `beam_size=1` matched `greedy_decode` on 100/100 sentences (checked before the input fix). Validation, first 200, with markers: beam 4 BLEU **41.08** (length ratio 0.971) vs. greedy **40.39**; paired bootstrap difference mean 0.86, range -0.29 to 2.41, beam 4 higher in 90.4% of resamples. **Not proven.** Beam 4 on the full test set with markers: **unmeasured**. Beam size and alpha not tuned; beam 2 unmeasured. |
  | Weeks 5–6 — other improvements | 🔴 Not Started | |
  | Weeks 7–8 — serving / Docker / tracking | 🔴 Not Started | |
  | Source `<sos>/<eos>` markers: training vs. decoding | ✅ Fixed and measured | Known Issues #11. Training-serving skew, fixed by one shared `encode_source` in `dataset.py`. Validation, first 200, greedy: 26.01 without markers, 40.39 with. Test, greedy, with markers: 37.58 |
  | AddNorm dropout placement vs. paper | ⚠️ Needs Verification | See Known Issues #1. Did not stop the real run from reaching ppl 5.71, so not urgent, but still an unresolved mismatch with the paper |
  | Full-size model training behavior (memory, speed, convergence) | ✅ Verified | Completed 40 configured epochs on Colab without the earlier divergence (after warmup 1000→4000, epochs 30→40) |

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
  - `overfit_sanity_check.py` — trains a **small** model (d_model 128, 2 layers, 1.7M params) on 10 real pairs for 500 steps. Passed on 2026-09-13 (loss 78.6 → ~0) and again on 2026-09-24 after the embedding-init fix (loss 9.6 → 0.003, faster convergence); near-perfect reproduction of all 10 sentences.

  ---

  ## 4. Current Phase

  **Phase 1 (Reproduction) → Weeks 3 and 4 complete (training, decoding, evaluation); first Phase 2 experiment (beam search) measured on 200 validation sentences only.** Week 3 finished 2026-09-29; Week 4 started 2026-10-01.

  ### Weeks 3 and 4 — result summary

  | Measure | Value |
  |---|---|
  | Training | `warmup_steps` 1000 made training **diverge around epochs 4 to 8**; fixed with `warmup_steps` 4000 and 40 epochs |
  | Best checkpoint | epoch 33, val loss **1.7422**, perplexity **5.71** (`experiments/best_model.pt`) |
  | Test, 1,000 sentences, German input WITH `<sos>/<eos>` (matches training) | no-model baseline **0.48** (unaffected by the input format) · greedy **37.58** · beam 4 **unmeasured** |

  Test set, greedy, German input with markers (1,000 sentences):

  | Metric | Greedy |
  |---|---|
  | BLEU | 37.58 (67.7/44.9/30.6/21.4) |
  | Length ratio (hyp/ref) | 1.025 |
  | Hard loops (same word 3+ in a row) | 3.0% |
  | Repeated 3-word phrase | 5.1% |
  | Ended without `<eos>` | 4 |
  | Decode time | 515 s (0.52 s per sentence) |

  Validation split, first 200 sentences (tuning only, not the final test numbers):

  | Run | BLEU |
  |---|---|
  | Greedy, German input without markers | 26.01 |
  | Greedy, German input with markers | 40.39 |
  | Beam 4 (alpha 0.6), with markers | 41.08 (length ratio 0.971) |

  Paired bootstrap, beam 4 vs. greedy, with markers (1000 resamples, seed 42): mean BLEU difference 0.86, 95% range -0.29 to 2.41, beam 4 higher in 90.4% of resamples. **Not proven:** the range includes zero.

  Glue rule check (`src/check_glue_rule.py`; validation, first 200 sentences, greedy; hyphen and apostrophe fixes applied to both outputs): old single-id rule BLEU **40.58** (BP 0.990), 57 unseen words of 2,364 (**2.4%**); new previous-id rule BLEU **40.88** (BP 0.995), 53 of 2,372 (**2.2%**); 20 of 200 outputs differ. The +0.30 was **not tested for significance**. In the agent's reading of 10 differing examples, 6 were better and 4 worse (the new rule writes "woman ' s"); that is a **judgment**, not a measurement. **Decision: keep the old rule.**

  **Decision: training stops.** Best checkpoint epoch 33, val loss 1.7422. A tokenizer retrain is deferred as an optional later experiment. Next phase: Phase 3.

  The beam 4 validation run was done in two parts because the laptop battery died: sentences 0-149, then 150-199 with `--start 150 --limit 50`. The join was checked: 200 entries, no duplicates, references match the validation set in order. The no-`<eos>` count for beam 4 cannot be recomputed from the text files.

  Beam 4 with markers, validation sentences 150-199: **482 s, 9.63 s per sentence** (50 sentences only; the time of the first 150 was lost when the battery died). Full-test beam 4 with markers: about 2.7 hours, **ESTIMATED** from 9.63 s per sentence, **not run**.

  **Measured with the wrong input format** (measured with German input without `<sos>/<eos>`; training used them; superseded, do not cite). Test set, 1,000 sentences: greedy BLEU 24.34, beam 4 (alpha 0.6) 30.17. Greedy to beam 4: length ratio 1.481 to 1.258, hard loops 13.2% to 6.7%, repeated 3-word phrase 29.3% to 21.2%, ended without `<eos>` 33 to 11, time about 13 minutes to 3.2 hours (11.53 s per sentence). First-200-test-sentence beam table, BLEU 24.06 / 30.00 / 32.27 for beam 1 / 2 / 4. `analyze_outputs.py` on the greedy output: 39.4% repeated 3-word sequence, 14.5% same-word run of 3+, 5.7% longer than the longest reference (real references: 0.7%, 0.0%).

  Known limits (still open): heuristic detokenizer (for example it produces "beigerug" -- Known Issues #2); beam size and alpha not tuned; beam search not batched (Known Issues #8); `best_model.pt` backup not confirmed (Known Issues #9); results are on Multi30k image captions only (Known Issues #12). **Unmeasured:** beam 4 on the full test set with markers; full-test BLEU for beam 2 or any other beam size/alpha; any measured breakdown of beam-4 errors by type (only a 40-sentence hand read exists, on no-marker output -- judgments, see Detail).

  ### Detail

  - **Week 3 summary (complete):**
    - `lr_schedule.py`, `label_smoothing.py`, the `training:` config, and `train.py` (train/val loop, warmup LR, label smoothing, checkpointing, `--smoke` mode) were all built and locally verified — see Week 3 entry under Completed Work.
    - **First full-size attempt diverged** (around epochs 4 to 8) with `warmup_steps=1000`/`num_epochs=30`; retuned to `warmup_steps=4000`/`num_epochs=40` (closer to the paper's original value) and re-run successfully. This corrects my earlier recommendation to shorten warmup below the paper's value — see Technical Changes and Important Decisions.
    - **Real training run completed on Colab:** best checkpoint at epoch 33/40, **validation loss 1.7422, perplexity 5.71**, saved to `experiments/best_model.pt` (580MB; gitignored, pulled back to this machine manually).
  - **`src/decode.py` built and verified (2026-10-01):** greedy autoregressive decoding (encode once, feed `<sos>`, predict one token at a time until `<eos>` or `max_len`). Same `sys.path` fix as `train.py` was needed (Known Issues #4). Tested on 3 real German sentences against `best_model.pt`: all 3 correctly stop at `<eos>` (confirmed by direct inspection, not just `max_len` truncation) and get the core meaning right, but **greedy decoding repeats itself** on 2 of 3 (`bushes bushes bushes bushes`, `house house house`) — a well-known failure mode of greedy search, not a code bug. This is exactly why beam search is already on the Week 5-6 roadmap (see Known Issues #7).
  - **BPE-rejoin investigated and addressed with a heuristic (2026-10-03):** inspecting `data/bpe_tokenizer.json` directly confirmed the vocab has no boundary marker at all (no `##`, `Ġ`, or `</w>` on any entry) — the tokenizer was trained with `BpeTrainer`'s defaults (no `continuing_subword_prefix`/`end_of_word_suffix`), so which pieces start a new word is genuinely not recoverable from a generated id sequence, not just a missing-decoder-object problem. The real fix (retrain the tokenizer with a boundary marker + retrain the whole model against the new vocab) would invalidate the current `best_model.pt`, so — discussed with the user and decided explicitly — `src/data/detokenize.py` instead builds a per-token-id "usually starts a word vs. usually continues one" table from the training corpus and uses it to rejoin generated ids. Tested locally on a small fixed sentence set: correctly reassembles split words (`weis`+`se`→`weisse`), with one known residual artifact (a stray space before punctuation, since `Whitespace` pre-tokenization treats punctuation as its own "word"). Retraining the tokenizer+model remains open as real Phase 2 follow-up if this heuristic proves insufficient at full scale — see Known Issues #2 and Important Decisions.
  - **`src/evaluate.py` built (2026-10-03), run on the laptop CPU (2026-10-04):** loads `best_model.pt`, builds the boundary map from the train split, greedy-decodes all 1,000 test sentences one at a time, rejoins with `clean_decode`, scores with `sacrebleu` against the Week 1 baseline (0.48), and saves hypotheses+references to `experiments/test_translations.txt`. Result at the time: BLEU 24.34 (measured with German input without `<sos>/<eos>`; training used them; superseded, see the note above). The run printed `Using device: cpu` and saved to `D:\Translator\experiments\test_translations.txt`. A first attempt on Colab failed because `best_model.pt` was not there.
  - **`src/analyze_outputs.py` built and run (2026-10-04):** re-scores `test_translations.txt`, tests whether the known stray-space-before-punctuation artifact (Known Issues #2) affects BLEU (it does not — `sacrebleu`'s own tokenizer absorbs it; score identical before/after the fix), and quantifies greedy-decoding repetition (Known Issues #7) against a same-metric baseline computed on the real references. Those percentages were measured on no-marker output and are in the note above. This turned "repetition looked bad in 2 of 3 decode.py examples" into a measured, dataset-wide number, which turned out to be mostly an input-format effect (Known Issues #7).
  - **Beam search implemented (2026-10-07, committed b246ca3):** `beam_search_decode` in `src/decode.py` (sum of log-probs, finished-beam pool, length normalization by `(generated tokens)^alpha`; one decoder pass per beam, no batching, no KV cache — slow on CPU). `config.yaml → decoding:` holds `beam_size: 4`, `length_alpha: 0.6`; `evaluate.py` takes `--beam-size`/`--alpha` overrides and writes `experiments/test_translations_beam{N}.txt` (the greedy `test_translations.txt` is never overwritten). `fix_punctuation_spacing` is now applied inside `clean_decode` (`src/data/detokenize.py`), so readable output no longer has the space before punctuation; BLEU is unaffected (see Known Issues #2).
  - **Beam verification (2026-10-07, `src/check_beam.py`, re-runnable):** (1) `beam_size=1` vs `greedy_decode` on the first 100 test sentences: **100/100 identical id lists** — beam implementation is consistent with greedy. (2) Beam-size table on the first 200 test sentences (alpha 0.6, CPU). It was run on no-marker input, so its numbers are in the note above only, and it is one of the two test-set uses recorded in the evaluation rule. Beam time grows much faster than linearly in beam size because each beam is a separate unbatched decoder pass with no cache (Known Issues #8).
  - **Full-test beam 1 run and `evaluate.py` changes (2026-10-08):**
    - **Weights-only checkpoint:** created `experiments/best_model_weights.pt` from `best_model.pt` (epoch 33, val loss 1.7422); `best_model.pt` untouched.
    - **Full beam-1 run:** `python src/evaluate.py --beam-size 1` on all 1,000 test sentences (local CPU, no-marker input), identical to the greedy result of that time. Confirms at full scale that `beam_size=1` == greedy. Numbers are in the note above. Output: `experiments/test_translations_beam1.txt` (old format).
    - **`evaluate.py` now supports partial and crash-safe runs** (pushed by the user): `--start S` / `--limit N` select sentences S to S+N; each translation is appended to the results file and flushed as soon as it is produced; partial runs write `experiments/test_translations_beam{B}_start{S}_limit{N}.txt` and print `PARTIAL RUN: sentences S to E only` at top and bottom; every run prints seconds per sentence; a results file that already exists is never overwritten (the run exits before decoding). Scoring code and `beam_search_decode` unchanged. Smoke-tested on `--start 5 --limit 3`, including the refusal on re-run.
    - Results files for runs with markers carry a `_markers` suffix, so the old no-marker files are never overwritten.
  - **Full-test beam 4 run, no-marker input (recorded 2026-10-15; superseded):** `experiments/test_translations_beam4.txt` has 1,000 HYP/REF entries and its references match the test set in order (checked 2026-10-15; the beam-1 file's references match too). Re-scored with `load_pairs`/`score` from `analyze_outputs.py`; the numbers are in the note above. It ran on the laptop CPU in one process, with results appended to the file as it ran. A beam 4 run on the full test set with markers has **not** been done.
  - **40-sentence hand read (2026-10-15) -- judgments from one random sample (`random.seed(42)`, 40 of the 1,000 test sentences), NOT a measurement.** Labelled by hand, on **no-marker outputs**, so it may not describe outputs made with the correct input. Beam 4 vs. greedy: **better 14, same 21, worse 5.** Tags (sentences carrying the tag), greedy then beam 4: runaway repeat 9 then 4; phrase repeat 23 then 18; garbled rare word 8 then 8; other wrong content 17 then 16; no tag at all 5 then 7. Reading: beam 4 mainly cut the worst runaway repeats; garbled rare words did not change. With 40 sentences and one labeller these counts have wide uncertainty. These test sentences were looked at to form the next hypotheses, which the evaluation rule in Important Decisions now forbids doing again.
  - **Input-format bug found and fixed.** Cause: training-serving skew. `TranslationDataset` wrapped the German source as `[sos] + ids + [eos]` for training, but `decode.py`, `evaluate.py` and `check_beam.py` encoded it without markers. Fix (commit `2008819`): one shared function, `encode_source` in `src/data/dataset.py`, now used by training and all three decoding scripts. An equality check against what `TranslationDataset` builds passed on the first 200 training sentences. `evaluate.py` result files now carry `_markers`; the `--split test|validation` option (file names `{split}_translations_beam...`, banner `SPLIT: validation (tuning only)`) was committed in `a6a1dfa`.
  - **Effect of the fix (measured):** validation, first 200, greedy: BLEU 26.01 without markers, 40.39 with (`src/check_src_markers.py`). Test, 1,000 sentences, greedy, with markers: BLEU 37.58, ratio 1.025, hard loops 3.0%, repeated 3-word phrase 5.1%, 4 ended without `<eos>`, 515 s. Why the gap is that large was not analysed. Beam 4 with markers was run on validation only (two parts, bootstrap): see the tables above. **Unmeasured:** beam 4 on the full test set with markers.
  - **Remaining in this phase:** qualitative read-through to characterize *what kinds* of sentences still repeat or get garbled with beam 4 (length? rare words? specific source constructions?).
  - **Known blockers:** none. The AddNorm dropout question (Known Issues #1) remains open but did not prevent a good training result, so it is no longer gating progress — worth revisiting as a controlled experiment later rather than before decoding.

  > Note: commit messages `14c8220` ("the warmup stage") and `732779e` ("added label smoothing to transformer") understated what changed at the time; `245d5ff`, `3c42935`, `d9c4eeb`, `de28311`, `916286b` carried the rest of Week 3 to completion.

  ---

  ## 5. Next Steps

  1. **Phase 3 (production):** FastAPI endpoint (German in, English out), Docker, experiment tracking, README. Serving must follow the `encode_source` / `clean_decode` rule in Important Decisions. Training stops; the glue-rule check is done and the old rule is kept.
  2. **Beam 4 is on hold** (not proven, about 19x slower than greedy -- see START HERE). Beam 4 on the full test set with markers: not run.
  3. **Deferred, optional later experiments:** a tokenizer retrain with a proper boundary marker and a model retrain against it (Known Issues #2); the AddNorm dropout placement (Known Issues #1); a warmup comparison against 4000; an alpha test (not decided).

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
  | 2026-09-24 | Added `training:` section to `config.yaml` | Batch size, epochs, warmup, smoothing, clip, Adam settings, seed now live in config, not code. |
  | 2026-09-24 | `TokenEmbedding` init: `nn.init.normal_(weight, std=d_model**-0.5)` | PyTorch's default N(0,1) × √d_model made embeddings ≈23× the positional encoding and the tied output logits huge. Initial loss at full size fell 344.6 → 12.9 (128/2: 107.7 → 9.9); the sanity check now starts at 9.6 (was 78.6) and converges faster. One shared tensor, so it covers encoder, decoder and output projection. Supersedes the earlier explanation that embeddings "start small". |
  | 2026-09-24 | `config.yaml → smoke_test:` + `train.py --smoke` | Lets the whole training pipeline be checked on this CPU with the small model and 256 pairs. |
  | 2026-09-24 | `src/train.py` adds `src/models` to `sys.path` | Model files use bare imports (`from decoder import …`) so they run standalone; without this, `from models.transformer import …` fails with `ModuleNotFoundError`. |
  | 2026-09-27 | Retuned `warmup_steps: 1000→4000`, `num_epochs: 30→40` | First full-size Colab run **diverged** at `warmup_steps=1000`. Reverting most of the way to the paper's 4000 (plus more epochs to still get useful post-warmup training) fixed it. This corrects the earlier reasoning in this file that a shorter warmup would suffice for a smaller dataset — see Important Decisions. |
  | 2026-09-29 | Checkpoint save path settled on `ROOT / "experiments"` | Briefly tried saving to `/content/drive/MyDrive/...` directly from the training loop; reverted to the local (Colab-runtime) `experiments/` path, with the resulting file pulled back to this machine manually afterward — `experiments/` stays gitignored. |
  | 2026-09-29 | **Real full-size training run completed on Colab** | 40 configured epochs, best checkpoint at epoch 33: **validation loss 1.74, perplexity 5.71**. `experiments/best_model.pt` (580MB) is local only, not committed. |
  | 2026-10-03 | Added `src/data/detokenize.py` (BPE-rejoin heuristic) and `src/evaluate.py` | Built in this order because `evaluate.py` needs readable rejoined text to produce a meaningful BLEU score (Known Issues #2) — see Important Decisions for why a heuristic was chosen over retraining the tokenizer. |
  | 2026-10-07 | Added `beam_search_decode`, `decoding:` config section, `evaluate.py` beam flags, punctuation fix inside `clean_decode`; added `src/check_beam.py` | First Phase 2 experiment (hypothesis: beam search cuts the greedy repetition in Known Issues #7). Verified equal to greedy at beam 1; beam-size table partially measured — see Current Phase. |
  | 2026-10-08 | `evaluate.py`: `--start/--limit`, incremental append+flush saving, partial-run file naming and banner, seconds/sentence, never-overwrite guard; created `experiments/best_model_weights.pt` | Long CPU/Colab beam runs can now be chunked and survive crashes without losing finished sentences. Full beam-1 run matched greedy (no-marker input). |
  | 2026-10-15 | Full-test beam 4 (alpha 0.6) run recorded on **no-marker input** (superseded); file integrity and re-scoring checked | Numbers are in the Section 4 note. No model, decoding or scoring code changed. |
  | 2026-10-04 | **`evaluate.py` run on the laptop CPU** (first evaluated result, no-marker input, superseded); added `src/analyze_outputs.py` | The analysis script showed the punctuation-spacing artifact (Known Issues #2) doesn't move the BLEU score at all, and measured repetition (Known Issues #7, numbers in the Section 4 note). |
  | 2026-10-18 (recorded) | **Source-marker bug fixed: `encode_source` in `dataset.py`** (commit `2008819`); `evaluate.py` gets `_markers` file names and `--split` (committed in `a6a1dfa`) | Training-serving skew: training wrapped the German source in `<sos>/<eos>`, decoding did not. Validation, first 200, greedy: 26.01 without markers, 40.39 with. Test greedy with markers: 37.58. All earlier BLEU/repetition numbers are superseded. |
  | 2026-10-18 (recorded) | Beam 4 vs. greedy on 200 validation sentences, with markers: 41.08 vs. 40.39 | Two-part run (laptop battery died), join checked. Paired bootstrap: mean difference 0.86, range -0.29 to 2.41, beam 4 higher in 90.4% of resamples. Not proven. |

  No new third-party dependencies since `requirements.txt` was created (torch, datasets, sacrebleu, pyyaml, tokenizers).

  ---

  ## 7. Known Issues / Technical Debt

  | # | Problem | Location | State | Blocks work? |
  |---|---|---|---|---|
  | 1 | ⚠️ **AddNorm applies dropout *after* LayerNorm** (`dropout(norm(x + sublayer))`). The paper (Sec. 5.4) applies dropout to the sub-layer output *before* the residual add and normalization. It is a locked "Post-LN" design either way, so this is a placement detail, not Pre-LN. The real training run reached perplexity 5.71 with this as-is, so it is not broken — but it is still a mismatch with the paper worth a controlled before/after comparison. | `src/models/add_norm.py` | Unresolved, non-blocking | No |
  | 2 | Decoding leaves BPE pieces unjoined (`play house`, `li on`). Root cause confirmed 2026-10-03 (not just "no decoder set"): the vocab itself has no boundary marker (`BpeTrainer` was run with default `continuing_subword_prefix`/`end_of_word_suffix`, i.e. none), so a generated id sequence genuinely cannot tell which pieces start a new word. Worked around with a heuristic (`src/data/detokenize.py`). Known residual artifact: stray space before punctuation (`something .`) — confirmed present in the real greedy test output and flagged by `sacrebleu`'s own "forgot to detokenize" warning, but **measured (2026-10-04, `analyze_outputs.py`) to have zero effect on the BLEU score itself** (`sacrebleu`'s internal tokenizer normalizes it away); it only affects how readable the raw text looks. (`tokenizer.decode()` does correctly strip `<sos>/<eos>/<pad>` by default — confirmed 2026-10-01.) | `src/data/tokenizer.py` (root cause), `src/data/detokenize.py` (workaround) | **Heuristic workaround implemented and validated**; true fix (retrain tokenizer with a boundary marker + retrain model) is open Phase 2 work, now lower priority since it doesn't move BLEU | No longer blocks Week 4 BLEU — confirmed via a real run (BLEU 24.34 at the time, measured with German input without `<sos>/<eos>`; training used them; superseded, Section 4 note) |
  | 3 | Source-side conventions differ: `dataset.py` wraps source with `<sos>/<eos>`, the sanity-check script does not. `collate_fn` also returns masks that `Transformer` ignores (it derives its own from `pad_id`). The same kind of mismatch between training and decoding is fixed -- see #11; the sanity-check script still does not wrap. | `src/data/dataset.py`, `src/models/overfit_sanity_check.py`, `src/models/transformer.py` | Unresolved | No; settle when writing `train.py` |
  | 4 | Model files import each other by bare name, so they only import cleanly from inside `src/models/`. `train.py` and `decode.py` both work around it with the same `sys.path.insert`; a later `evaluate.py`/API will need it too, or the models should become a proper package (which would break their standalone `__main__` runs). | `src/models/*.py`, `src/train.py`, `src/decode.py` | Worked around (twice now) | No |
  | 5 | `lr_schedule.py` has inconsistent 8-space indentation and no imports (unused, harmless). Cosmetic. | `src/training/lr_schedule.py` | Runs correctly | No |
  | 6 | `README.md` is a 2-line stub. | `README.md` | Deferred to Phase 3 polish | No |
  | 7 | **Repetition in greedy output: most of what was measured earlier came from the input bug (#11), not from the model.** The old percentages (no-marker input) are in the Section 4 note only. With the correct input, test greedy (1,000 sentences, `evaluate.py` definitions) has hard loops (same word 3+ in a row) **3.0%**, repeated 3-word phrase **5.1%**, 4 ended without `<eos>`. Repetition is much lower but not gone. Whether beam 4 reduces it further with markers is **unmeasured** on the test set; on 200 validation sentences the beam 4 BLEU gain is not proven (bootstrap, #10). (`analyze_outputs.py` and `evaluate.py` use different definitions of repetition -- do not mix them.) | `src/decode.py`, `src/evaluate.py`, model behavior | Mostly explained by the input bug; remainder open | No |
  | 8 | **Beam search is not batched and has no KV cache** (one decoder pass per beam per step, one sentence at a time). In the old no-marker run the full test set at beam 4 took ~3.2 hours (11.53 s per sentence) vs. ~13 minutes for greedy; greedy with markers took 515 s. Beam 4 time on the full test set with markers is unmeasured. Limits how many beam sizes/alphas can be tried. | `src/decode.py` | Open | Slows tuning, nothing else |
  | 9 | **`best_model_weights.pt` is backed up** (confirmed by the user); it is not in git. `best_model.pt` (552.8 MiB, with optimizer state) is only needed to resume training, and its backup is **not confirmed**. | `experiments/` | weights backed up; full checkpoint not | No |
  | 10 | **Beam size (4) and length alpha (0.6) are not tuned.** They are config defaults. With markers, only greedy (= beam 1) has a full-test BLEU (37.58); beam 4 was measured on 200 validation sentences only. | `configs/config.yaml → decoding:` | Open | No |
  | 11 | ✅ **Fixed and measured -- source `<sos>/<eos>` markers differed between training and decoding (training-serving skew).** `TranslationDataset` wrapped the German source as `[sos] + ids + [eos]` for training (so the validation loss 1.74 was measured with the markers); `decode.py`, `evaluate.py` and `check_beam.py` encoded it without them (the tokenizer has no post-processor). Fixed by one shared function, `encode_source` in `src/data/dataset.py`, used by training and all decoding scripts (commit `2008819`). Measured on validation, first 200, greedy: BLEU 26.01 without markers, 40.39 with (`src/check_src_markers.py`). Test, greedy, with markers: 37.58. Old no-marker numbers are in the Section 4 note. Why the gap is so large was not analysed. | `src/data/dataset.py`, `src/decode.py`, `src/evaluate.py`, `src/check_beam.py` | Fixed | No |
  | 12 | **Known limit: all results are on Multi30k image captions only** (short, clean, about 12 words). How the model does on other kinds of German text is not measured. | whole project | Known limit | No |

  Environment gotchas (not bugs):
  - The Bash tool's terminal mangles UTF-8 (umlauts show as `�`); use PowerShell to view German text. Data on disk is correct.
  - Scripts that import `datasets` print a harmless `ResourceTracker … _recursion_count` traceback at exit on Windows.
  - `HF_TOKEN` warning and occasional `getaddrinfo failed` retries are harmless once the dataset is cached.
  - Mounting Google Drive in Colab fails for the user with "credential propagation was unsuccessful" (a restart and a fresh notebook did not help). Workaround: call `files.download()` on the checkpoint the moment a run finishes.
  - A Colab runtime reset deleted the checkpoint of the first completed 40-epoch training run before it was downloaded, so the training was repeated. Download results as soon as a run ends.

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
  | `overfit_sanity_check.py` | ✅ Re-verified | Passed 2026-09-13; re-run 2026-09-24 after the init fix: loss 9.56 → 0.003, sentences reproduced |
  | `training/lr_schedule.py` | ✅ Re-verified | Matches reference values at steps 1, 2000, 4000, 16000 |
  | `training/label_smoothing.py` | ✅ Re-verified | Demo output matches expectation (gap 0.54 vs 0.10) |
  | `src/train.py --smoke` | ✅ Tested and working (local, tiny) | 2026-09-24: 15 epochs on 256 pairs with the 1.7M-param model; val loss 54.6 → 5.80; LR warmup and checkpoint save/keys correct |
  | `src/train.py` (full size, Colab) | ✅ Tested and working | 2026-09-29: 40 configured epochs, best checkpoint epoch 33, val loss 1.74 (ppl 5.71); checkpoint has all 5 expected keys (verified locally 2026-10-01) |
  | Initial-loss / embedding-init fix | ✅ Tested and working | 2026-09-24, 64 real pairs, eval mode: full-size initial loss 344.6 → 12.9; small 107.7 → 9.9 (see Technical Changes) |
  | `tokenizer.decode()` special-token stripping | ✅ Tested and working | 2026-10-01: default call already drops `<sos>/<eos>/<pad>`; no extra code needed in `decode.py` |
  | `decode.py` / greedy decoding | ✅ Tested and working | 2026-10-01, 3 real sentences: all reach `<eos>` correctly (verified directly, not assumed); core meaning right; repetition artifacts present (Known Issues #7) |
  | `data/detokenize.py` (BPE-rejoin heuristic) | ✅ Tested locally + validated at scale | 2026-10-03, 3 fixed sentences: correctly rejoins split words (`weis`+`se`→`weisse`). 2026-10-04: ran across the real 1,000-sentence greedy output (`test_translations.txt`); residual stray-space-before-punctuation artifact confirmed present but shown not to affect BLEU |
  | `evaluate.py` | ✅ Tested and working (laptop CPU) | 2026-10-04: full run against `best_model.pt` and the 1,000-sentence test set completed (BLEU 24.34, measured with German input without `<sos>/<eos>`; training used them; superseded). With markers, greedy on the test set: BLEU 37.58. The `--split` change and `_markers` file names were used for the validation runs; committed in `a6a1dfa` |
  | `analyze_outputs.py` | ✅ Tested and working | 2026-10-04: run against the real `test_translations.txt`; output: 1000/1000 pairs loaded, BLEU unchanged by the punctuation fix; repetition stats were on no-marker output (Section 4 note) |
  | `evaluate.py` `--start/--limit`, incremental save, overwrite guard | ✅ Tested and working (small) | 2026-10-08: `--beam-size 1 --start 5 --limit 3` ran, banner/seconds-per-sentence/file name correct; identical re-run refused. Full beam-1 run (1,000 sentences, no-marker input) also completed with the new code path and matched greedy. Crash-mid-run behavior reasoned from the flush-per-sentence code, not simulated |
  | `beam_search_decode` (`decode.py`) | ✅ Tested and working | 2026-10-07: `beam_size=1` == `greedy_decode` on 100/100 sentences. 2026-10-08: full 1,000-sentence beam 1 matched greedy (no-marker input). 2026-10-15: a full-test beam 4 run on no-marker input was re-scored independently (superseded, Section 4 note). With markers: beam 4 on 200 validation sentences only (BLEU 41.08, two-part run, join checked). Beam 4 on the full test set with markers, full-test beam 2 and other alphas **unmeasured** |
  | `check_beam.py` | ✅ Tested and working (CPU, ~50 min total) before the input fix | Re-runnable correctness check + beam-size table; now uses `encode_source` but has **not been re-run** since |
  | `encode_source` (`dataset.py`) | ✅ Tested and working | Equality check against what `TranslationDataset` builds on the first 200 training sentences passed, so training behaviour is unchanged |
  | `check_src_markers.py` | ✅ Run | Validation, first 200, greedy: 26.01 without markers, 40.39 with |
  | `check_beam_gain.py`, `score_results.py` (not yet committed) | ✅ Run | Join check passed (200 entries, no duplicates, references match the validation set in order); paired bootstrap, 1000 resamples, seed 42: difference 0.86, range -0.29 to 2.41, beam 4 higher in 90.4% |
  | Test-set BLEU (trained model vs. 0.48 baseline) | ✅ Measured | Greedy **37.58** with markers vs. 0.48 baseline. Beam 4 on the full test set with markers: **unmeasured**. |

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
  - **Evaluation rule (set 2026-10-15): tune on the VALIDATION split, report on the test split.** From now on anything that is tuned or chosen -- beam size, alpha, repeat rules, decoding changes -- is decided on validation sentences; the test split is used only for final reported numbers. **Recorded honestly:** test sentences have already been used twice for decisions: (1) the first-200-test-sentence beam-size table in `check_beam.py` (beam 4 was chosen from it), and (2) the 40-sentence hand read. The old test scores (24.34, 30.17) are superseded (measured with German input without `<sos>/<eos>`; training used them). Any future beam 4 test number is not fully clean of test-set influence, because beam 4 was chosen using the first-200-test-sentence table.
  - **All German model input goes through `encode_source` (set 2026-10-18).** Training and every decoding script build the source with the one shared function in `src/data/dataset.py`, so the input format cannot drift apart again. Results files written with it carry `_markers`; old no-marker files are kept and never overwritten.
  - **Serving rule (set for Phase 3):** serving must build German input with `encode_source` and clean the output with the same `clean_decode` as `evaluate.py`, and a test must check the ids match training. No such test exists yet.
  - **Warmup steps: 4000 (the paper's value), not a shortened warmup, and 40 epochs.** This file previously argued 4000 was tuned for WMT-scale data and recommended ~1000 for 29k pairs; that was tried and the model **diverged**. Reverting close to the paper's original value fixed it. Recorded here so the earlier (wrong) reasoning isn't repeated: dataset size alone isn't a safe basis for cutting warmup — treat it as a value to test, not assume.
  - **BPE-rejoin: heuristic workaround, not a tokenizer retrain (decided 2026-10-03, with the user).** Direct inspection of `data/bpe_tokenizer.json` showed the vocab has no word-boundary marker on any entry, so no decoder object can correctly rejoin a generated id sequence — the information needed was discarded during tokenizer training (`BpeTrainer` with default `continuing_subword_prefix`/`end_of_word_suffix`), not missing from the decode step. The correct fix — retrain the tokenizer with a boundary marker and retrain the model against the new vocab — would invalidate `best_model.pt` (val loss 1.74) and cost a new Colab run before any BLEU number exists. Given that cost, chose instead to approximate word boundaries from training-corpus statistics (`src/data/detokenize.py`) so `evaluate.py` can produce a real, if imperfect, BLEU number now. **Validated 2026-10-04:** the heuristic's residual error (stray space before punctuation) measurably exists but does not move the BLEU score (`sacrebleu` normalizes it internally — confirmed via `analyze_outputs.py`'s before/after comparison, identical score). This lowers the urgency of a tokenizer retrain — it would improve human readability, not the evaluation metric — so it stays optional Phase 2 follow-up rather than a priority.

  References: [CLAUDE.md](CLAUDE.md) (goals, hardware, agent rules, build plan by week), `configs/config.yaml` (all hyperparameters), the paper's Sections 3–5 (architecture, Sec. 5.3 LR schedule, Sec. 5.4 regularization).

  ---

  Last Updated: 2026-10-18
