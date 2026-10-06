import argparse
import sys
import time
from pathlib import Path

import sacrebleu
import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# Same reason as train.py/decode.py: the model files import each other by
# bare name (`from decoder import ...`), which only resolves with
# src/models on sys.path.
sys.path.insert(0, str(ROOT / "src" / "models"))

from analyze_outputs import has_repeated_ngram, longest_run
from data.detokenize import build_boundary_map, clean_decode
from decode import beam_search_decode
from models.transformer import Transformer


def parse_args(decoding_cfg):
    # Defaults come from config.yaml -> decoding, so the CLI only overrides.
    parser = argparse.ArgumentParser(description="Decode the test set and score it with sacrebleu.")
    parser.add_argument("--beam-size", type=int, default=decoding_cfg["beam_size"])
    parser.add_argument("--alpha", type=float, default=decoding_cfg["length_alpha"])
    return parser.parse_args()


def main():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args = parse_args(config["decoding"])
    beam_size, alpha = args.beam_size, args.alpha

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | beam_size={beam_size} | alpha={alpha}")

    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    special = config["tokenizer"]["special_tokens"]
    PAD_ID = tokenizer.token_to_id(special["pad"])
    SOS_ID = tokenizer.token_to_id(special["sos"])
    EOS_ID = tokenizer.token_to_id(special["eos"])
    special_ids = {PAD_ID, SOS_ID, EOS_ID}

    model = Transformer(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
    ).to(device)

    checkpoint = torch.load(ROOT / "experiments" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss {checkpoint['best_val_loss']:.4f}")

    dataset_name = config["dataset"]["name"]
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]

    # The tokenizer's vocab carries no marker for "this piece starts a new
    # word" (Known Issues #2) -- ids the model generates one at a time don't
    # come with that info attached, so it's approximated from how each id
    # behaved in the training corpus. See src/data/detokenize.py for why.
    print("Building subword boundary map from the training corpus...")
    train_raw = load_dataset(dataset_name, split="train")
    train_sentences = list(train_raw[src_lang]) + list(train_raw[tgt_lang])
    boundary_map = build_boundary_map(tokenizer, train_sentences)
    print(f"Boundary map covers {len(boundary_map)} token ids")

    test_set = load_dataset(dataset_name, split="test")

    hypotheses = []
    references = []
    ended_without_eos = 0

    print(f"Decoding {len(test_set)} test sentences with beam search (one at a time)...")
    start = time.time()
    for i, example in enumerate(test_set):
        german_sentence = example[src_lang]
        reference = example[tgt_lang]

        src_ids = tokenizer.encode(german_sentence).ids
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != PAD_ID).unsqueeze(1)

        output_ids, finished = beam_search_decode(
            model, src, src_mask,
            max_len=config["model"]["max_len"],
            sos_id=SOS_ID, eos_id=EOS_ID, device=device,
            beam_size=beam_size, alpha=alpha,
        )
        if not finished:
            ended_without_eos += 1

        translation = clean_decode(output_ids, tokenizer, boundary_map, special_ids)

        hypotheses.append(translation)
        references.append(reference)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_set)}  ({time.time() - start:.0f}s)")
    elapsed = time.time() - start

    n = len(hypotheses)
    hard_loops = sum(longest_run(h.split()) >= 3 for h in hypotheses)
    repeated_phrases = sum(has_repeated_ngram(h.split(), 3) for h in hypotheses)

    print(f"\nTest set size: {n}")
    print(sacrebleu.corpus_bleu(hypotheses, [references], force=True))
    print(f"hard loops (same word 3+ times in a row): {hard_loops} ({100 * hard_loops / n:.1f}%)")
    print(f"repeated 3-word phrase:                   {repeated_phrases} ({100 * repeated_phrases / n:.1f}%)")
    print(f"ended without <eos>:                      {ended_without_eos}")
    print(f"time taken:                               {elapsed:.0f}s")

    # Beam results go to their own file. The greedy baseline,
    # experiments/test_translations.txt, is never written by this script.
    results_path = ROOT / "experiments" / f"test_translations_beam{beam_size}.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for hyp, ref in zip(hypotheses, references):
            f.write(f"HYP: {hyp}\nREF: {ref}\n\n")
    print(f"Saved hypotheses + references to {results_path}")


if __name__ == "__main__":
    main()
