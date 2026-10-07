import sys
import time
from pathlib import Path

import sacrebleu
import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))

from analyze_outputs import has_repeated_ngram, longest_run
from data.detokenize import build_boundary_map, clean_decode
from decode import beam_search_decode, greedy_decode
from models.transformer import Transformer

ALPHA = 0.6


def main():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    special = config["tokenizer"]["special_tokens"]
    PAD_ID = tokenizer.token_to_id(special["pad"])
    SOS_ID = tokenizer.token_to_id(special["sos"])
    EOS_ID = tokenizer.token_to_id(special["eos"])
    special_ids = {PAD_ID, SOS_ID, EOS_ID}
    max_len = config["model"]["max_len"]

    model = Transformer(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=max_len,
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
    ).to(device)
    checkpoint = torch.load(ROOT / "experiments" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"device={device} | epoch {checkpoint['epoch']} | val_loss {checkpoint['best_val_loss']:.4f}")

    ds_name = config["dataset"]["name"]
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]

    train_raw = load_dataset(ds_name, split="train")
    boundary_map = build_boundary_map(tokenizer, list(train_raw[src_lang]) + list(train_raw[tgt_lang]))
    test_set = load_dataset(ds_name, split="test")

    def encode(i):
        ids = tokenizer.encode(test_set[i][src_lang]).ids
        src = torch.tensor([ids], dtype=torch.long, device=device)
        return src, (src != PAD_ID).unsqueeze(1)

    # ── 1. beam_size=1 vs greedy_decode ──
    diffs = []
    for i in range(100):
        src, src_mask = encode(i)
        g = greedy_decode(model, src, src_mask, max_len, SOS_ID, EOS_ID, device)
        b, _ = beam_search_decode(model, src, src_mask, max_len, SOS_ID, EOS_ID, device,
                                  beam_size=1, alpha=ALPHA)
        if g != b:
            diffs.append((i, g, b))
    print(f"\n=== Check 1: beam_size=1 vs greedy ===\nmatches: {100 - len(diffs)}/100")
    for i, g, b in diffs[:3]:
        print(f"[{i}] greedy: {g}\n     beam1:  {b}")

    # ── 2. table ──
    n = 200
    refs = [test_set[i][tgt_lang] for i in range(n)]
    print(f"\n=== Check 2: first {n} test sentences, alpha={ALPHA} ===")
    print(f"{'beam':>4} {'BLEU':>6} {'len_ratio':>9} {'loops%':>7} {'rep3%':>6} {'no_eos':>6} {'secs':>6}")
    for beam_size in (1, 2, 4):
        hyps, no_eos = [], 0
        start = time.time()
        for i in range(n):
            src, src_mask = encode(i)
            ids, finished = beam_search_decode(model, src, src_mask, max_len, SOS_ID, EOS_ID, device,
                                               beam_size=beam_size, alpha=ALPHA)
            no_eos += not finished
            hyps.append(clean_decode(ids, tokenizer, boundary_map, special_ids))
        secs = time.time() - start
        bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True)
        loops = 100 * sum(longest_run(h.split()) >= 3 for h in hyps) / n
        reps = 100 * sum(has_repeated_ngram(h.split(), 3) for h in hyps) / n
        print(f"{beam_size:>4} {bleu.score:>6.2f} {bleu.sys_len / bleu.ref_len:>9.3f} "
              f"{loops:>7.1f} {reps:>6.1f} {no_eos:>6} {secs:>6.0f}")


if __name__ == "__main__":
    main()
