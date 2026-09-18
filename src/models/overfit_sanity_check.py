from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

from transformer import Transformer


# ── 1. SETUP ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))

special   = config["tokenizer"]["special_tokens"]
PAD_ID    = tokenizer.token_to_id(special["pad"])
SOS_ID    = tokenizer.token_to_id(special["sos"])
EOS_ID    = tokenizer.token_to_id(special["eos"])


# ── 2. DATA — first 10 pairs only ─────────────────────────────────────────────

ds         = load_dataset(config["dataset"]["name"], split="train")
src_lang   = config["dataset"]["source_lang"]
tgt_lang   = config["dataset"]["target_lang"]
NUM_PAIRS  = 10

def encode_pair(example):
    src_ids = tokenizer.encode(example[src_lang]).ids
    tgt_ids = tokenizer.encode(example[tgt_lang]).ids

    src = torch.tensor(src_ids, dtype=torch.long)
    tgt_input = torch.tensor([SOS_ID] + tgt_ids, dtype=torch.long)
    tgt_target = torch.tensor(tgt_ids + [EOS_ID], dtype=torch.long)

    return src, tgt_input, tgt_target

pairs = [encode_pair(ds[i]) for i in range(NUM_PAIRS)]

def pad_and_stack(sequences, pad_id):
    return pad_sequence(sequences, batch_first=True, padding_value=pad_id)

src_list        = [p[0] for p in pairs]
tgt_input_list  = [p[1] for p in pairs]
tgt_target_list = [p[2] for p in pairs]

src        = pad_and_stack(src_list,       PAD_ID).to(device)
tgt_input  = pad_and_stack(tgt_input_list, PAD_ID).to(device)
tgt_target = pad_and_stack(tgt_target_list, PAD_ID).to(device)

print(f"src shape:        {src.shape}")
print(f"tgt_input shape:  {tgt_input.shape}")
print(f"tgt_target shape: {tgt_target.shape}")


# ── 3. MODEL ──────────────────────────────────────────────────────────────────

model = Transformer(
    vocab_size = config["tokenizer"]["vocab_size"],
    d_model    = config["sanity_check"]["d_model"],
    num_heads  = config["sanity_check"]["num_heads"],
    d_ff       = config["sanity_check"]["d_ff"],
    num_layers = config["sanity_check"]["num_layers"],
    max_len    = config["model"]["max_len"],
    dropout    = config["model"]["dropout"],
    pad_id     = PAD_ID,
).to(device)

model.train()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── 4. OPTIMIZER AND LOSS ─────────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)


# ── 5. TRAINING LOOP ──────────────────────────────────────────────────────────

NUM_STEPS = 500

for step in range(NUM_STEPS):
    optimizer.zero_grad()

    logits = model(src, tgt_input)  # [B, S_tgt, vocab_size]

    logits_flat = logits.reshape(-1, logits.size(-1))  # [B * S_tgt, vocab_size]
    targets_flat = tgt_target.reshape(-1)              # [B * S_tgt]

    loss = criterion(logits_flat, targets_flat)

    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if step % 10 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")


# ── 6. VERIFICATION ───────────────────────────────────────────────────────────

print("\n--- Sanity Check Result ---")

assert loss.item() < 0.1, f"Final loss {loss.item():.4f} did not drop below 0.1 — something is likely wrong upstream"

model.eval()
with torch.no_grad():
    logits = model(src, tgt_input)
    predicted_ids = logits.argmax(dim=-1)  # [B, S_tgt]

    for i in range(NUM_PAIRS):
        german_sentence = ds[i][src_lang]
        english_actual = ds[i][tgt_lang]
        english_predicted = tokenizer.decode(predicted_ids[i].tolist())

        print(f"\nGerman:    {german_sentence}")
        print(f"Predicted: {english_predicted}")
        print(f"Actual:    {english_actual}")
