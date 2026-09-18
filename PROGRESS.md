# Project Progress & Architecture

This file is the onboarding doc: read this top to bottom and you should understand what this
project is, how the pieces fit together, and what's been built so far — without needing to dig
through commit history or ask anyone.

For the project's goals, constraints, and working style (why decisions are made the way they
are), see [CLAUDE.md](CLAUDE.md). This file tracks the *build* — what exists, how it flows
end-to-end, and a dated log of what changed and why.

**How this file is maintained:** it gets updated whenever a meaningful change is made to the
project (new component, decision made, structure change) — not on every line of code, but on
every step that changes how the system works or what exists.

> **Note on this update (2026-09-17):** this file was last kept current on 2026-08-18, covering
> only the very first day of work (data exploration + tokenizer). Everything built since then —
> the rest of Week 1 and all of Week 2 — was never logged here. The Progress Log below has been
> reconstructed from git commit history to bring this file back in sync with the actual state of
> the repo. From this point forward, it is updated with every meaningful change as it happens.

---

## What this project is

A German→English machine translation system, built by implementing the Transformer
architecture ("Attention Is All You Need") from scratch in PyTorch — no pretrained models, no
HuggingFace `pipeline()`. Trained on the Multi30k dataset (~29k German/English sentence pairs).

## Architecture — end-to-end flow

**Weeks 1 and 2 are complete.** Every stage below is built and verified. What's not yet
built is the real training run and evaluation (Week 3 onward) — see "What's next."

```
Multi30k (HuggingFace dataset)
        │
        │  train split only (29,000 pairs) — validation/test held out
        │  to avoid leaking their vocabulary into the tokenizer
        ▼
combined_corpus.txt
  one German+English sentence per line, both languages interleaved
  in a single file
        │
        ▼
BPE tokenizer training (tokenizers library)
  - single shared tokenizer for BOTH languages (not one per language)
  - vocab size: 8000 subword tokens
  - special tokens reserved: <pad>(0) <sos>(1) <eos>(2) <unk>(3)
        │
        ▼
bpe_tokenizer.json  (the trained, reusable tokenizer artifact)
        │
        ▼
No-model baseline (src/baseline.py)
  "translate" by copying German unchanged, score against real English with sacrebleu
  → BLEU = 0.48 — the floor every real result must beat
        │
        ▼
PyTorch Dataset / DataLoader (src/data/dataset.py)
  - TranslationDataset: tokenizes each pair once upfront, wraps with <sos>/<eos>
  - __getitem__ splits target into decoder_input ([sos, w1..wn]) and
    decoder_target ([w1..wn, eos]) BEFORE padding, on the raw sequence
  - collate_fn: pads source/decoder_input/decoder_target per-batch, builds
    padding masks (True = real token, False = padding)
        │
        ▼
Transformer (src/models/) — built from scratch, matches the paper
  - TokenEmbedding: lookup table, scaled by √d_model
  - PositionalEncoding: fixed sin/cos pattern added to embeddings, + dropout
  - Encoder: 6 × EncoderBlock (self-attention → add&norm → feed-forward → add&norm)
  - Decoder: 6 × DecoderBlock (masked self-attention → add&norm →
    cross-attention against encoder output → add&norm → feed-forward → add&norm)
  - Output projection: Linear(d_model → vocab_size), weight-tied to the
    embedding table (encoder embedding = decoder embedding = projection weight)
  - Masking: source padding mask; target mask = causal mask × padding mask
        │
        ▼
Overfit sanity check (src/models/overfit_sanity_check.py)  ✅ PASSED
  Small model (d_model=128, 2 layers, 1.7M params — full 512-dim/6-layer model
  is too slow to overfit on local CPU) trained on 10 real sentence pairs for
  500 steps. Loss: 78.6 → ~0. Model reproduces all 10 sentences almost
  perfectly. Proves the entire architecture + training wiring is correct.
        │
        ▼  ← NOT YET BUILT (Week 3+)
Training loop (hand-written, LR warmup + label smoothing)
  full-size model, full 29k dataset, on Google Colab GPU
        │
        ▼
Decoding (greedy → later beam search)
  autoregressive: feed <sos>, predict next token, feed it back, repeat until <eos>
        │
        ▼
English output text  (+ real BLEU score via sacrebleu against test set references,
compared against the 0.48 baseline)
```

Eventually (Phase 3), this becomes a live "request flow" too: a FastAPI endpoint takes German
text in, runs it through the same tokenizer → model → decoder pipeline, and returns English
text out, containerized with Docker.

## Repository structure (what exists right now)

```
Translator/
├── CLAUDE.md                # project charter: goals, phases, hardware constraints, working style
├── PROGRESS.md               # this file — architecture + build log
├── .gitignore                 # excludes /data/, /experiments/, venv, caches, secrets
├── .vscode/
│   └── settings.json          # workspace auto-save enabled
├── requirements.txt           # torch, datasets, sacrebleu, pyyaml, tokenizers
├── configs/
│   └── config.yaml            # all hyperparameters/settings — nothing hardcoded in code.
│                                # dataset, tokenizer (vocab_size, special tokens), model
│                                # (real d_model=512/6-layer dims) and a separate small
│                                # sanity_check section used only for local CPU verification
├── data/                      # gitignored — generated artifacts, not source
│   ├── combined_corpus.txt      # generated: raw text corpus used to train the tokenizer
│   └── bpe_tokenizer.json       # generated: the trained BPE tokenizer
├── src/
│   ├── explore.py              # Week 1 data inspection: loads Multi30k, prints sample pairs
│   ├── baseline.py             # no-model baseline BLEU score (0.48)
│   ├── data/
│   │   ├── tokenizer.py          # builds combined_corpus.txt, trains the shared BPE tokenizer
│   │   └── dataset.py            # TranslationDataset + collate_fn (Dataset/DataLoader)
│   └── models/
│       ├── embeddings.py         # TokenEmbedding, PositionalEncoding
│       ├── attention.py          # scaled_dot_product_attention
│       ├── multi_head_attention.py  # MultiHeadAttention
│       ├── feed_forward.py       # FeedForward (position-wise)
│       ├── add_norm.py           # AddNorm (residual connection + LayerNorm)
│       ├── encoder.py            # EncoderBlock, Encoder (6-layer stack)
│       ├── decoder.py            # generate_causal_mask, DecoderBlock, Decoder (6-layer stack)
│       ├── transformer.py        # Transformer — wires Encoder + Decoder + output projection,
│       │                           weight tying, mask generation
│       └── overfit_sanity_check.py  # trains a tiny model on 10 examples to verify correctness
└── experiments/                # gitignored — reserved for future training logs/checkpoints
```

`src/train.py` and `src/evaluate.py` (the real training loop and evaluation script) don't exist
yet — they land in Week 3 and Week 4.

**Note:** every file in this project is a plain `.py` module now. `explore.ipynb` and
`tokenizer.ipynb` were both converted to `.py` for consistency, clean git diffs, and easy
importing between files (see "Key decisions" below).

## How to get this running from scratch (for a new machine / new agent)

```powershell
cd D:\Translator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then, from the project root, with the venv active:

```powershell
python src\explore.py                        # inspect the raw dataset
python src\data\tokenizer.py                 # regenerates data/combined_corpus.txt and
                                              # data/bpe_tokenizer.json (both gitignored)
python src\baseline.py                       # no-model baseline BLEU
python src\data\dataset.py                   # Dataset/DataLoader self-test
python src\models\embeddings.py              # TokenEmbedding + PositionalEncoding self-test
python src\models\attention.py               # scaled dot-product attention self-test
python src\models\multi_head_attention.py    # MultiHeadAttention self-test
python src\models\feed_forward.py            # FeedForward self-test
python src\models\add_norm.py                # AddNorm self-test
python src\models\encoder.py                 # full Encoder self-test
python src\models\decoder.py                 # full Decoder self-test
python src\models\transformer.py             # full Transformer + weight-tying check
python src\models\overfit_sanity_check.py    # trains a tiny model on 10 examples (~1 min on CPU)
```

Every architecture file under `src/models/` is runnable on its own and prints shape/sanity
checks — this is how each piece was verified as it was built, and it's the fastest way for a
newcomer to confirm the environment is set up correctly.

## Key decisions made so far (and why)

- **Tokenization: BPE, not word-level.** German is highly compounding (e.g.
  `Antriebsradsystem`), so word-level tokenization either treats compounds as rare/OOV tokens
  or blows up vocab size. BPE breaks them into reusable subword pieces instead. Matches the
  original paper's approach.
- **Shared vocabulary (one tokenizer for both languages), not separate per-language
  tokenizers.** Matches the paper's approach; keeps the pipeline simpler. It's also what makes
  weight tying (below) possible.
- **Vocab size: 8000, not the paper's ~37k.** The paper's vocab size was tuned for WMT-scale
  data (millions of pairs). With only 29k pairs here, a vocab that large would mean most tokens
  are seen only a handful of times each, producing poorly-trained embeddings — effectively
  defeating the point of subword tokenization. 8000 keeps merges to genuinely frequent, reusable
  subword units.
- **Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`.** Needed because this is sequence
  generation, not classification: `<sos>`/`<eos>` mark where a sequence starts/stops so the
  decoder knows when to stop generating; `<pad>` lets variable-length sentences be batched
  together; `<unk>` is the fallback for anything the tokenizer wasn't trained on.
- **Tokenizer trained only on the `train` split.** Validation/test sentences are held out from
  tokenizer training to avoid leaking their vocabulary in — keeps evaluation honest.
- **Decoder input/target shift happens before padding, on the raw sequence.** `__getitem__`
  slices `[sos, w1..wn, eos]` into `decoder_input = [sos, w1..wn]` and
  `decoder_target = [w1..wn, eos]` per-example, then `collate_fn` pads each independently.
  Doing the shift after padding (slicing a padded batch tensor) can still work numerically once
  padding is excluded from the loss, but shifting first is unambiguous and is the standard,
  easy-to-reason-about approach.
- **Weight tying.** The encoder's token embedding, the decoder's token embedding, and the final
  output projection all share the exact same weight tensor (`encoder_embedding is
  decoder_embedding is output_projection.weight` → `True`). Cuts parameter count and forces the
  model to learn one consistent notion of what a token means, rather than three independent
  ones.
- **Two separate model-size configs.** `configs/config.yaml` has a real `model:` section
  (d_model=512, 6 layers — the paper's actual size, used for real training on Colab) and a
  separate, much smaller `sanity_check:` section (d_model=128, 2 layers) used only by
  `overfit_sanity_check.py`. The full-size model is far too slow to overfit locally on a CPU
  with no GPU; the small config exists purely to verify correctness quickly.
- **Everything is `.py`, not `.ipynb`.** Notebooks were used briefly for early exploration
  (`explore.ipynb`, `tokenizer.ipynb`) but both were converted to plain scripts once the
  actual workflow settled on terminal execution rather than interactive cell-by-cell work.
  Plain `.py` gives clean git diffs, painless imports between files, and matches the
  production-engineering habits this project is meant to build toward.
- **No `Co-Authored-By` line in git commits for this repo** (documented in `CLAUDE.md`) —
  commits should read as the user's own work regardless of default attribution behavior.

## Progress log

### 2026-08-19
- Repo scaffolded: `configs/`, `data/`, `src/`, `src/data/`, `src/models/`, `experiments/`,
  `.gitignore`, `requirements.txt`.
- Data inspection (`src/explore.py`, originally a notebook): loaded Multi30k via `datasets`,
  confirmed split sizes (train 29,000 / validation 1,014 / test 1,000), spot-checked 10 training
  examples — clean pairs, correct German encoding, no obvious anomalies in the sample.
- Decided tokenization strategy: BPE, shared German+English vocab, size 8000.
- Built the tokenizer training script (originally `tokenizer.ipynb`): combines the train split's
  German + English sentences into `data/combined_corpus.txt`, trains a shared BPE tokenizer via
  the `tokenizers` library (`Whitespace` pre-tokenizer, `BpeTrainer`), saves to
  `data/bpe_tokenizer.json`. Verified vocab size is exactly 8000 and encoding output looks
  correct (e.g. `vieler` → `viel`+`er`, `Büsche` → `Bü`+`sche`).
- Added `tokenizer` section to `configs/config.yaml`.
- Enabled VS Code workspace auto-save; `.gitignore` extended to exclude env files/keys/credentials.

### 2026-08-21
- Built `src/baseline.py`: the no-model baseline. "Translates" by copying the German test
  sentences unchanged and scoring them against the real English references with `sacrebleu`.
  Result: **BLEU = 0.48** on the 1,000-sentence test set — the floor every trained result must
  beat.

### 2026-08-22
- Built `src/data/dataset.py`: `TranslationDataset` (tokenizes all pairs once upfront, wraps
  with `<sos>`/`<eos>`) and `collate_fn` (per-batch padding + padding masks).
- Fixed a real bug: `.gitignore`'s `data/` rule (no leading slash) was matching *any* folder
  named `data` anywhere in the repo, silently excluding `src/data/` from git entirely —
  `dataset.py` and `tokenizer.py` had never actually been tracked. Fixed by anchoring the rule
  to the repo root (`/data/`).
- Added the decoder input/target shift (`decoder_input`/`decoder_target`, split before padding)
  to `TranslationDataset.__getitem__` — needed for teacher-forcing training in Week 3, and
  cleaner to get right now while the Dataset code was still fresh.

### 2026-08-24 – 2026-08-28
- Built `src/models/embeddings.py`: `TokenEmbedding` (lookup table scaled by `√d_model`) and
  `PositionalEncoding` (fixed sin/cos pattern, added to embeddings, with dropout applied per the
  paper). Added `model` section to `config.yaml` (`d_model`, `max_len`, `dropout`).
- Converted `explore.ipynb` to `explore.py` (verified identical output) — first step toward an
  all-`.py` codebase.

### 2026-09-01
- Built `src/models/attention.py` (`scaled_dot_product_attention`) and
  `src/models/multi_head_attention.py` (`MultiHeadAttention`, 8 heads). Verified masking works
  correctly (padded positions get exactly `0.0` attention weight, remaining weights renormalize
  to sum to 1).

### 2026-09-07
- Built `src/models/feed_forward.py` (`FeedForward`, `d_ff=2048`) and `src/models/add_norm.py`
  (`AddNorm` — residual connection + `LayerNorm` + dropout).

### 2026-09-09
- Built `src/models/encoder.py`: `EncoderBlock` (self-attention → add&norm → feed-forward →
  add&norm) and `Encoder` (embeddings → positional encoding → 6 stacked `EncoderBlock`s via
  `nn.ModuleList`). Verified: token IDs `[2, 10]` → `[2, 10, 512]`.

### 2026-09-12
- Built `src/models/decoder.py`: `generate_causal_mask`, `DecoderBlock` (masked self-attention →
  add&norm → cross-attention against encoder output → add&norm → feed-forward → add&norm), and
  `Decoder` (6 stacked `DecoderBlock`s). Verified with fake encoder output of a *different*
  sequence length than the target — confirms cross-attention correctly handles Q and K/V having
  different lengths.

### 2026-09-13
- Built `src/models/transformer.py`: `Transformer` wires `Encoder` + `Decoder` + output
  projection together, with weight tying across encoder embedding / decoder embedding / output
  projection. Verified: `src [2,10]` + `tgt [2,8]` → `logits [2,8,8000]`; weight tying confirmed
  with `is` checks.
- Converted `tokenizer.ipynb` to `src/data/tokenizer.py` (verified identical output) — codebase
  is now fully `.py`.
- Built the first version of `src/models/overfit_sanity_check.py` (training loop over 10 real
  sentence pairs, teacher forcing, gradient clipping).

### 2026-09-15
- Re-ran the overfit sanity check using the full-size model config (`d_model=512`, 6 layers) —
  found it was far too slow to complete locally on CPU (no GPU on this machine). Added a
  dedicated `sanity_check` section to `config.yaml` with a much smaller model (`d_model=128`,
  2 layers, ~1.7M params) used only by this script, keeping the real `model:` section untouched
  for the actual Colab training run.
- **Overfit sanity check passed:** loss dropped from `78.6` to effectively `0` over 500 steps;
  the model correctly reproduced nearly all 10 memorized sentences (remaining artifacts —
  repeated trailing words past `<eos>`, and BPE subword pieces not being rejoined on decode —
  are expected/cosmetic, not correctness issues; see script output).
- **Week 2 complete.** Every architecture component from the paper (embeddings through output
  projection) is built, connected end-to-end, and proven to actually learn.

## What's next

**Week 3 — Training loop.** Hand-written training loop (no high-level `Trainer` abstractions)
using the full-size model: the paper's learning-rate warmup schedule, label smoothing, and a
full training run on all 29,000 Multi30k training pairs — run on Google Colab's GPU, since this
is well beyond what the local CPU can handle in reasonable time. After that: Week 4's greedy
decoding and real BLEU evaluation against the 0.48 baseline.
