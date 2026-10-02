import sys
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

from data.detokenize import build_boundary_map, clean_decode
from decode import greedy_decode  # reuse the function already built and verified
from models.transformer import Transformer

BASELINE_BLEU = 0.48  # src/baseline.py: copy German unchanged, scored on this same test set


def main():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Decoding {len(test_set)} test sentences (one at a time -- this will take a while)...")
    for i, example in enumerate(test_set):
        german_sentence = example[src_lang]
        reference = example[tgt_lang]

        src_ids = tokenizer.encode(german_sentence).ids
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != PAD_ID).unsqueeze(1)

        output_ids = greedy_decode(
            model, src, src_mask,
            max_len=config["model"]["max_len"],
            sos_id=SOS_ID, eos_id=EOS_ID, device=device,
        )
        translation = clean_decode(output_ids, tokenizer, boundary_map, special_ids)

        hypotheses.append(translation)
        references.append(reference)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_set)}")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"\nTest set size: {len(test_set)}")
    print(f"Model BLEU:    {bleu.score:.2f}")
    print(f"Baseline BLEU: {BASELINE_BLEU:.2f}")
    print(f"Improvement:   {bleu.score - BASELINE_BLEU:+.2f}")

    # Worth keeping for error analysis later, not just the one final number.
    results_path = ROOT / "experiments" / "test_translations.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for hyp, ref in zip(hypotheses, references):
            f.write(f"HYP: {hyp}\nREF: {ref}\n\n")
    print(f"Saved hypotheses + references to {results_path}")


if __name__ == "__main__":
    main()
