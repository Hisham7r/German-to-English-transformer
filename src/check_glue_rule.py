import random
import sys
from collections import Counter
from pathlib import Path

import sacrebleu
import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# Same reason as evaluate.py/decode.py: the model files import each other by
# bare name, which only resolves with src/models on sys.path.
sys.path.insert(0, str(ROOT / "src" / "models"))

from check_text_fixes import fix_apostrophe, fix_hyphen
from check_unseen_words import words
from data.dataset import encode_source
from data.detokenize import build_boundary_map, clean_decode, fix_punctuation_spacing
from decode import greedy_decode
from models.transformer import Transformer

N = 200
MIN_PAIR_COUNT = 3


def build_pair_votes(tokenizer, sentences):
    """For each (previous id, this id) pair in the TRAIN corpus: how often did
    this id start a new pre-tokenizer word, and how often did it continue one?
    The first token of a sentence has no previous id and is skipped."""
    starts, continues = Counter(), Counter()
    for encoding in tokenizer.encode_batch(list(sentences)):
        ids, word_ids = encoding.ids, encoding.word_ids
        for k in range(1, len(ids)):
            pair = (ids[k - 1], ids[k])
            if word_ids[k] != word_ids[k - 1]:
                starts[pair] += 1
            else:
                continues[pair] += 1
    return starts, continues


def clean_decode_pairs(ids, tokenizer, boundary_map, pair_starts, pair_continues, special_ids):
    """Like clean_decode, but a (previous id, this id) pair seen at least
    MIN_PAIR_COUNT times in train decides by its majority; otherwise the old
    single-id rule decides. The previous id is the previous non-special id."""
    pieces = []
    prev = None
    for token_id in ids:
        if token_id in special_ids:
            continue
        s, c = pair_starts[(prev, token_id)], pair_continues[(prev, token_id)]
        if prev is not None and s + c >= MIN_PAIR_COUNT:
            starts_word = s >= c
        else:
            starts_word = boundary_map.get(token_id, True)
        token_str = tokenizer.id_to_token(token_id)
        if pieces and not starts_word:
            pieces[-1] += token_str
        else:
            pieces.append(token_str)
        prev = token_id
    return fix_punctuation_spacing(" ".join(pieces))


def apply_fixes(text):
    # Fix C from check_text_fixes.py: hyphen fix and apostrophe fix together.
    return fix_apostrophe(fix_hyphen(text))


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    config = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | greedy_decode | validation split, first {N} sentences")

    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    special = config["tokenizer"]["special_tokens"]
    PAD_ID, SOS_ID, EOS_ID = (tokenizer.token_to_id(special[k]) for k in ("pad", "sos", "eos"))
    special_ids = {PAD_ID, SOS_ID, EOS_ID}

    m = config["model"]
    model = Transformer(
        vocab_size=config["tokenizer"]["vocab_size"], d_model=m["d_model"], num_heads=m["num_heads"],
        d_ff=m["d_ff"], num_layers=m["num_layers"], max_len=m["max_len"], dropout=m["dropout"],
        pad_id=PAD_ID,
    ).to(device)
    checkpoint = torch.load(ROOT / "experiments" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss {checkpoint['best_val_loss']:.4f}")

    ds_name = config["dataset"]["name"]
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]

    # TRAIN split only, German + English, same corpus evaluate.py uses for the old rule.
    train = load_dataset(ds_name, split="train")
    train_sentences = list(train[src_lang]) + list(train[tgt_lang])
    train_english_vocab = set()
    for sentence in train[tgt_lang]:
        train_english_vocab.update(words(sentence))

    print("Building the old single-id boundary map and the new (previous id, id) pair counts from train...")
    boundary_map = build_boundary_map(tokenizer, train_sentences)
    pair_starts, pair_continues = build_pair_votes(tokenizer, train_sentences)
    print(f"single-id map covers {len(boundary_map)} ids | "
          f"pairs seen in train: {len(set(pair_starts) | set(pair_continues))}, "
          f"pairs with >= {MIN_PAIR_COUNT} sightings: "
          f"{sum(1 for p in set(pair_starts) | set(pair_continues) if pair_starts[p] + pair_continues[p] >= MIN_PAIR_COUNT)}")

    # 1. Decode once; keep the id lists in memory only.
    val = load_dataset(ds_name, split="validation")
    german = [val[i][src_lang] for i in range(N)]
    refs = [val[i][tgt_lang] for i in range(N)]

    print(f"Decoding {N} validation sentences once with greedy_decode...")
    id_lists = []
    for sentence in german:
        src = torch.tensor([encode_source(sentence, tokenizer, SOS_ID, EOS_ID)], dtype=torch.long, device=device)
        src_mask = (src != PAD_ID).unsqueeze(1)
        id_lists.append(greedy_decode(model, src, src_mask, m["max_len"], SOS_ID, EOS_ID, device))

    # 2 + 3. Same ids, two gluing rules; 4. same hyphen/apostrophe fixes on both.
    old_out = [apply_fixes(clean_decode(ids, tokenizer, boundary_map, special_ids)) for ids in id_lists]
    new_out = [apply_fixes(clean_decode_pairs(ids, tokenizer, boundary_map, pair_starts, pair_continues,
                                              special_ids)) for ids in id_lists]

    # 5. Scores.
    print("\n=== Results (hyphen + apostrophe fixes applied to both; only the glue rule differs) ===")
    for name, outs in (("old rule (single id)", old_out), ("new rule (previous id + id)", new_out)):
        bleu = sacrebleu.corpus_bleu(outs, [refs], force=True)
        total = unseen = 0
        for s in outs:
            ws = words(s)
            total += len(ws)
            unseen += sum(w not in train_english_vocab for w in ws)
        print(f"\n{name}\n  {bleu}\n  unseen words: {unseen}/{total} = {100 * unseen / total:.1f}%")

    differ = [i for i in range(N) if old_out[i] != new_out[i]]
    print(f"\nOutputs that differ between the two rules: {len(differ)}/{N}")

    # 6. Examples.
    print("\n=== 10 differing sentences (random.Random(42), shown in index order) ===")
    for i in sorted(random.Random(42).sample(differ, min(10, len(differ)))):
        print(f"\n[{i}]\nold: {old_out[i]}\nnew: {new_out[i]}\nREF: {refs[i]}")

    print("\nNote: deciding which of the two outputs is better in these examples is my judgment, "
          "not a measurement. BLEU and the unseen-word share are the only measured numbers here.")


if __name__ == "__main__":
    main()
