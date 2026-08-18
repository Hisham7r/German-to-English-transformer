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

---

## What this project is

A German→English machine translation system, built by implementing the Transformer
architecture ("Attention Is All You Need") from scratch in PyTorch — no pretrained models, no
HuggingFace `pipeline()`. Trained on the Multi30k dataset (~29k German/English sentence pairs).

## Architecture — end-to-end flow

The system is being built in stages (see CLAUDE.md's week-by-week plan). This section reflects
what's **actually implemented right now**, and will grow as later stages land.

### Current pipeline (data preparation stage)

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
  - special tokens reserved: <pad> <sos> <eos> <unk>
        │
        ▼
bpe_tokenizer.json  (the trained, reusable tokenizer artifact)
```

### Planned pipeline (not yet built)

```
bpe_tokenizer.json
        │
        ▼
PyTorch Dataset / DataLoader
  - tokenizes each (de, en) pair to token IDs
  - collate_fn pads batches to equal length, builds attention masks
        │
        ▼
Transformer (encoder-decoder, built from scratch)
  - token embeddings + positional encoding
  - encoder stack (self-attention + feed-forward)
  - decoder stack (self-attention + cross-attention + feed-forward)
  - output projection → vocab-size logits
        │
        ▼
Training loop (hand-written, LR warmup + label smoothing)
  trained on Google Colab GPU — this machine has no GPU
        │
        ▼
Decoding (greedy → later beam search)
  autoregressive: feed <sos>, predict next token, feed it back, repeat until <eos>
        │
        ▼
English output text  (+ BLEU score via sacrebleu against test set references)
```

Eventually (Phase 3), this becomes a live "request flow" too: a FastAPI endpoint takes German
text in, runs it through the same tokenizer → model → decoder pipeline, and returns English
text out, containerized with Docker.

## Repository structure (what exists right now)

```
Translator/
├── CLAUDE.md              # project charter: goals, phases, hardware constraints, working style
├── PROGRESS.md            # this file — architecture + build log
├── .gitignore              # excludes data/, experiments/, venv, caches
├── .vscode/
│   └── settings.json       # workspace auto-save enabled (so notebook cell outputs persist to
│                            # disk automatically — lets an agent read outputs/errors directly)
├── requirements.txt        # torch, datasets, sacrebleu, pyyaml, tokenizers
├── configs/
│   └── config.yaml         # all hyperparameters/settings — nothing hardcoded in code.
│                            # currently: dataset name/languages, tokenizer vocab_size + special tokens
├── data/                   # gitignored — generated/downloaded artifacts, not source
│   ├── combined_corpus.txt   # generated: raw text corpus used to train the tokenizer
│   └── bpe_tokenizer.json    # generated: the trained BPE tokenizer
├── src/
│   ├── explore.ipynb        # Week 1 data inspection: loads Multi30k, prints sample pairs,
│   │                         # confirms split sizes and clean encoding
│   └── data/
│       └── tokenizer.ipynb  # builds combined_corpus.txt from the train split, trains the
│                             # shared BPE tokenizer, saves it, sanity-checks encode/decode
└── experiments/            # gitignored — reserved for future training logs/checkpoints
```

`src/models/` (architecture code) and `src/train.py` / `src/evaluate.py` don't exist yet —
they land in later stages.

## How to get this running from scratch (for a new machine / new agent)

```powershell
cd D:\Translator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then open `src/explore.ipynb` and `src/data/tokenizer.ipynb` in VS Code, select the `.venv`
interpreter as the kernel, and run cells top to bottom. `tokenizer.ipynb` regenerates
`data/combined_corpus.txt` and `data/bpe_tokenizer.json` (both gitignored, so they don't exist
until you run it).

## Key decisions made so far (and why)

- **Tokenization: BPE, not word-level.** German is highly compounding (e.g.
  `Antriebsradsystem`), so word-level tokenization either treats compounds as rare/OOV tokens
  or blows up vocab size. BPE breaks them into reusable subword pieces instead. Matches the
  original paper's approach.
- **Shared vocabulary (one tokenizer for both languages), not separate per-language
  tokenizers.** Matches the paper's approach; keeps the pipeline simpler.
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

## Progress log

### 2026-08-18
- Repo scaffolded: `configs/`, `data/`, `src/`, `src/data/`, `src/models/`, `experiments/`,
  `.gitignore`, `requirements.txt`.
- Data inspection (`src/explore.ipynb`): loaded Multi30k via `datasets`, confirmed split sizes
  (train 29,000 / validation 1,014 / test 1,000), spot-checked 10 training examples — clean
  pairs, correct German encoding (umlauts/ß render correctly), no obvious anomalies in the
  sample.
- Decided tokenization strategy: BPE, shared German+English vocab, size 8000 (reasoning above).
- Built `src/data/tokenizer.ipynb`: combines the train split's German + English sentences into
  `data/combined_corpus.txt`, trains a shared BPE tokenizer via the `tokenizers` library
  (`Whitespace` pre-tokenizer, `BpeTrainer`), saves to `data/bpe_tokenizer.json`. Verified
  end-to-end via direct terminal execution (not just relying on notebook output) — confirmed
  vocab size is exactly 8000 and encoding output looks correct (e.g. `vieler` → `viel`+`er`,
  `Büsche` → `Bü`+`sche`), including a check for encoding corruption (there was none — an
  earlier run appeared to show mangled umlauts, but that turned out to be the Bash tool's
  terminal failing to render UTF-8, not real data corruption; confirmed via PowerShell and a raw
  hex dump of the file on disk).
- Added `tokenizer` section to `configs/config.yaml` (`vocab_size: 8000`, special tokens).
- Enabled VS Code workspace auto-save (`.vscode/settings.json`) so notebook cell outputs persist
  to disk without manual save — lets an agent read outputs/errors directly from the `.ipynb`
  file instead of requiring copy-paste.

## What's next

Per the Week 1 plan: build the PyTorch `Dataset` and `DataLoader`, including a `collate_fn` that
tokenizes each sentence pair with the trained BPE tokenizer, pads batches to equal length, and
produces attention masks. After that: the no-model baseline (Week 1's required "number to
beat"), then the model skeleton (Week 2).
