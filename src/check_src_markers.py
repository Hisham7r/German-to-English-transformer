import sys
import time
from pathlib import Path

import sacrebleu
import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# Same reason as evaluate.py: model files import each other by bare name.
sys.path.insert(0, str(ROOT / "src" / "models"))

from analyze_outputs import has_repeated_ngram, longest_run
from data.detokenize import build_boundary_map, clean_decode
from decode import greedy_decode
from models.transformer import Transformer

N = 200


def decode_all(model, tokenizer, boundary_map, special_ids, examples, src_lang,
               max_len, sos_id, eos_id, pad_id, device, wrap_markers):
    hypotheses = []
    ended_without_eos = 0
    start = time.time()
    for example in examples:
        src_ids = tokenizer.encode(example[src_lang]).ids
        if wrap_markers:
            src_ids = [sos_id] + src_ids + [eos_id]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != pad_id).unsqueeze(1)

        output_ids = greedy_decode(model, src, src_mask, max_len=max_len,
                                   sos_id=sos_id, eos_id=eos_id, device=device)
        if output_ids[-1] != eos_id:
            ended_without_eos += 1
        hypotheses.append(clean_decode(output_ids, tokenizer, boundary_map, special_ids))
    return hypotheses, ended_without_eos, time.time() - start


def report(name, hypotheses, references, ended_without_eos, seconds):
    n = len(hypotheses)
    hard_loops = sum(longest_run(h.split()) >= 3 for h in hypotheses)
    repeated = sum(has_repeated_ngram(h.split(), 3) for h in hypotheses)
    print(f"\n{name}")
    print(sacrebleu.corpus_bleu(hypotheses, [references], force=True))
    print(f"  hard loops:            {hard_loops} ({100 * hard_loops / n:.1f}%)")
    print(f"  repeated 3-word phrase: {repeated} ({100 * repeated / n:.1f}%)")
    print(f"  ended without <eos>:   {ended_without_eos}")
    print(f"  seconds:               {seconds:.0f}s")


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

    # Boundary map from the TRAIN split, exactly as evaluate.py does.
    train_raw = load_dataset(dataset_name, split="train")
    train_sentences = list(train_raw[src_lang]) + list(train_raw[tgt_lang])
    boundary_map = build_boundary_map(tokenizer, train_sentences)

    # VALIDATION split, first N sentences. The test split is not touched.
    val_raw = load_dataset(dataset_name, split="validation")
    examples = [val_raw[i] for i in range(N)]
    sources = [e[src_lang] for e in examples]
    references = [e[tgt_lang] for e in examples]
    max_len = config["model"]["max_len"]

    common = dict(model=model, tokenizer=tokenizer, boundary_map=boundary_map,
                  special_ids=special_ids, examples=examples, src_lang=src_lang,
                  max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
                  device=device)

    print(f"Decoding the first {N} validation sentences, greedy, version A (no markers)...")
    hyp_a, noeos_a, sec_a = decode_all(wrap_markers=False, **common)
    print(f"Decoding the same sentences, version B ([sos] + ids + [eos])...")
    hyp_b, noeos_b, sec_b = decode_all(wrap_markers=True, **common)

    report("A: no markers (as evaluate.py does today)", hyp_a, references, noeos_a, sec_a)
    report("B: [sos] + ids + [eos] (as TranslationDataset builds training source)",
           hyp_b, references, noeos_b, sec_b)

    differing = [i for i in range(N) if hyp_a[i] != hyp_b[i]]
    print(f"\ndiffer: {len(differing)}/{N}")
    for i in differing[:5]:
        print(f"\n[validation {i}]")
        print(f"  DE : {sources[i]}")
        print(f"  A  : {hyp_a[i]}")
        print(f"  B  : {hyp_b[i]}")
        print(f"  REF: {references[i]}")


if __name__ == "__main__":
    main()
