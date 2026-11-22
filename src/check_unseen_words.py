import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

BEAM1 = EXP / "validation_translations_beam1_start0_limit200_markers.txt"
BEAM4 = EXP / "validation_translations_beam4_start0_limit200_markers_joined.txt"

N = 200


def load_entries(path):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip("\n")
    hyps, refs = [], []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("HYP: "):
                hyps.append(line[len("HYP: "):])
            elif line.startswith("REF: "):
                refs.append(line[len("REF: "):])
    return hyps, refs


def is_punct(ch):
    return unicodedata.category(ch).startswith("P")


def words(sentence):
    # lowercase, split on whitespace, punctuation stripped from both ends;
    # tokens that are only punctuation disappear
    out = []
    for token in sentence.lower().split():
        start, end = 0, len(token)
        while start < end and is_punct(token[start]):
            start += 1
        while end > start and is_punct(token[end - 1]):
            end -= 1
        if end > start:
            out.append(token[start:end])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ds_name = config["dataset"]["name"]
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]

    train = load_dataset(ds_name, split="train")
    train_vocab = set()
    for sentence in train[tgt_lang]:
        train_vocab.update(words(sentence))
    print(f"English sentences in the train split: {len(train)}")
    print(f"distinct English words in the train split: {len(train_vocab)}")

    val = load_dataset(ds_name, split="validation")
    val_src = [val[i][src_lang] for i in range(N)]
    val_refs = [val[i][tgt_lang] for i in range(N)]

    hyp1, refs1 = load_entries(BEAM1)
    hyp4, refs4 = load_entries(BEAM4)
    print(f"\nbeam-1 entries: {len(hyp1)} | beam-4 entries: {len(hyp4)}")
    print(f"REF lines match the first {N} validation sentences: "
          f"beam-1 {'yes' if refs1 == val_refs else 'no'}, beam-4 {'yes' if refs4 == val_refs else 'no'}")
    if len(hyp1) != N or len(hyp4) != N or refs1 != val_refs or refs4 != val_refs:
        sys.exit("STOP: files do not match the first 200 validation sentences")

    print("\n=== Words not in the training set (first 200 validation sentences) ===")
    print(f"{'':<14}{'total words':>12}{'unseen':>8}{'share':>8}")
    unseen_counts = {}
    for name, sents in (("beam 1", hyp1), ("beam 4", hyp4), ("references", refs4)):
        total = 0
        counter = Counter()
        for s in sents:
            ws = words(s)
            total += len(ws)
            counter.update(w for w in ws if w not in train_vocab)
        unseen = sum(counter.values())
        unseen_counts[name] = counter
        print(f"{name:<14}{total:>12}{unseen:>8}{100 * unseen / total:>7.1f}%")

    for name in ("beam 4", "references"):
        print(f"\n20 most common unseen words in {name}:")
        for w, c in unseen_counts[name].most_common(20):
            print(f"  {c:>3}  {w}")

    print("\n=== 15 beam-4 sentences that contain unseen words ===")
    with_unseen = [i for i in range(N) if any(w not in train_vocab for w in words(hyp4[i]))]
    print(f"{len(with_unseen)}/{N} beam-4 sentences contain at least one unseen word; "
          "15 chosen at random (random.Random(42)), shown in index order")
    for i in sorted(random.Random(42).sample(with_unseen, min(15, len(with_unseen)))):
        bad = [w for w in words(hyp4[i]) if w not in train_vocab]
        print(f"\n[{i}]\nGerman:  {val_src[i]}\nbeam 4:  {hyp4[i]}\nREF:     {val_refs[i]}\nunseen:  {bad}")

    print("\nNote: whether an unseen word is a garbled fragment or a real rare word is my judgment, "
          "not a measurement. 'Unseen' only means 'not a whole word in the training English'.")
    print("Note: the detokenizer leaves spaces around hyphens in model output (e.g. 'long - sleeved'), "
          "so the model's hyphenated words are split into parts here, while the references keep them whole.")


if __name__ == "__main__":
    main()
