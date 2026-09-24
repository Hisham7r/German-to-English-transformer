import argparse
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

# The model files import each other by bare name (`from decoder import ...`) so
# each can also run standalone. That only resolves if src/models is on sys.path.
sys.path.insert(0, str(ROOT / "src" / "models"))

from data.dataset import TranslationDataset, collate_fn
from models.transformer import Transformer
from training.lr_schedule import get_warmup_lr_lambda


# ── SETUP ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="tiny local CPU pipeline check")
args = parser.parse_args()
SMOKE = args.smoke

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

train_cfg = config["training"]
smoke_cfg = config["smoke_test"]
torch.manual_seed(train_cfg["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (" | SMOKE TEST MODE" if SMOKE else ""))

tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
special = config["tokenizer"]["special_tokens"]
PAD_ID = tokenizer.token_to_id(special["pad"])
SOS_ID = tokenizer.token_to_id(special["sos"])
EOS_ID = tokenizer.token_to_id(special["eos"])

BATCH_SIZE = smoke_cfg["batch_size"] if SMOKE else train_cfg["batch_size"]
NUM_EPOCHS = smoke_cfg["num_epochs"] if SMOKE else train_cfg["num_epochs"]
WARMUP_STEPS = smoke_cfg["warmup_steps"] if SMOKE else train_cfg["warmup_steps"]
CKPT_NAME = "smoke_model.pt" if SMOKE else "best_model.pt"

model_dims = config["sanity_check"] if SMOKE else config["model"]
D_MODEL = model_dims["d_model"]


# ── DATA ─────────────────────────────────────────────────────────────────────

dataset_name = config["dataset"]["name"]
src_lang = config["dataset"]["source_lang"]
tgt_lang = config["dataset"]["target_lang"]

train_raw = load_dataset(dataset_name, split="train")
val_raw = load_dataset(dataset_name, split="validation")

if SMOKE:
    train_raw = train_raw.select(range(smoke_cfg["train_pairs"]))
    val_raw = val_raw.select(range(smoke_cfg["val_pairs"]))

train_set = TranslationDataset(train_raw, tokenizer, SOS_ID, EOS_ID, src_lang, tgt_lang)
val_set = TranslationDataset(val_raw, tokenizer, SOS_ID, EOS_ID, src_lang, tgt_lang)

collate = partial(collate_fn, pad_id=PAD_ID)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

print(f"Train pairs: {len(train_set):,} ({len(train_loader)} steps/epoch) | "
      f"Val pairs: {len(val_set):,} | batch size: {BATCH_SIZE}")


# ── MODEL, OPTIMIZER, SCHEDULER, CRITERION ────────────────────────────────────

model = Transformer(
    vocab_size=config["tokenizer"]["vocab_size"],
    d_model=D_MODEL,
    num_heads=model_dims["num_heads"],
    d_ff=model_dims["d_ff"],
    num_layers=model_dims["num_layers"],
    max_len=config["model"]["max_len"],
    dropout=config["model"]["dropout"],
    pad_id=PAD_ID,
).to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# base lr=1.0: get_warmup_lr_lambda returns the absolute LR, and LambdaLR multiplies it in
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=tuple(train_cfg["adam_betas"]),
    eps=train_cfg["adam_eps"],
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=get_warmup_lr_lambda(D_MODEL, WARMUP_STEPS)
)

train_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=train_cfg["label_smoothing"])
# No smoothing here: smoothing puts a floor under the loss, which would inflate perplexity.
val_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)


# ── TRAINING FUNCTION ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    # The loader also yields two padding masks from collate_fn. Ignore them (`_`):
    # Transformer.forward builds its own masks from pad_id, in the shapes it needs.
    for src, tgt_input, tgt_target, _, _ in loader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_target = tgt_target.to(device)

        optimizer.zero_grad()

        logits = model(src, tgt_input)  # [B, S_tgt, vocab_size]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip_norm"])
        optimizer.step()
        scheduler.step()

        # criterion averages over non-pad tokens, so weight by that count to get
        # a true per-token average across batches of different lengths
        n_tokens = (tgt_target != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


# ── EVALUATION FUNCTION ───────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt_input, tgt_target, _, _ in loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_target = tgt_target.to(device)

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

            n_tokens = (tgt_target != PAD_ID).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens


# ── MAIN TRAINING LOOP ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    best_val_loss = float("inf")
    print("(train loss is label-smoothed; val loss and perplexity are not — they are not comparable)")

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, train_criterion, device)
        val_loss = evaluate(model, val_loader, val_criterion, device)

        perplexity = math.exp(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        line = (
            f"epoch {epoch + 1:3d}/{NUM_EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"ppl {perplexity:9.2f} | lr {current_lr:.2e} | {time.time() - start:.0f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = ROOT / "experiments"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_dir / CKPT_NAME,
            )
            line += " | saved"

        print(line)
