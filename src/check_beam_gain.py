import random
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

BEAM1 = EXP / "validation_translations_beam1_start0_limit200_markers.txt"
BEAM4_FIRST = EXP / "validation_translations_beam4_start0_limit200_markers.txt"
BEAM4_LAST = EXP / "validation_translations_beam4_start150_limit50_markers.txt"
BEAM4_MERGED = EXP / "validation_translations_beam4_start0_limit200_markers_joined.txt"

N = 200
BOOTSTRAP_RUNS = 1000


def load_entries(path):
    """Returns a list of (hyp, ref); ref is None for an entry with HYP but no REF line."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip("\n")
    entries = []
    for block in text.split("\n\n"):
        hyp = ref = None
        for line in block.split("\n"):
            if line.startswith("HYP: "):
                hyp = line[len("HYP: "):]
            elif line.startswith("REF: "):
                ref = line[len("REF: "):]
        entries.append((hyp, ref))
    return entries


def fail(msg):
    sys.exit(f"STOP: {msg}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("Files used:")
    for p in (BEAM1, BEAM4_FIRST, BEAM4_LAST, BEAM4_MERGED):
        print(f"  {p.name}")

    # ── 1. check the join ──
    beam1 = load_entries(BEAM1)
    first = load_entries(BEAM4_FIRST)
    last = load_entries(BEAM4_LAST)
    merged = load_entries(BEAM4_MERGED)

    print("\n=== 1. Join check ===")
    for name, entries in (("beam-1 file", beam1), ("beam-4 first partial", first),
                          ("beam-4 150-199", last), ("beam-4 merged", merged)):
        print(f"entries in {name}: {len(entries)}")

    cut_off = first[-1][0] is not None and first[-1][1] is None
    print(f"first partial file has a cut-off last entry (HYP but no REF): {'yes' if cut_off else 'no'}")
    incomplete = [name for name, entries in (("beam-1", beam1), ("first", first), ("150-199", last), ("merged", merged))
                  if any(h is None or r is None for h, r in entries)]
    if incomplete:
        fail(f"entries missing HYP or REF in: {incomplete}")

    first_ok = merged[:150] == first[:150]
    last_ok = merged[150:200] == last
    count_ok = len(merged) == N
    dupes = len(merged) - len(set(merged))
    print(f"merged[0:150] == first 150 entries of first file: {'yes' if first_ok else 'no'}")
    print(f"merged[150:200] == whole 150-199 file:            {'yes' if last_ok else 'no'}")
    print(f"merged has exactly {N} entries:                    {'yes' if count_ok else 'no'}")
    print(f"duplicate (HYP, REF) entries in merged:           {dupes}")
    if not (first_ok and last_ok and count_ok and dupes == 0):
        fail("the merged file does not match its parts")

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]
    val = load_dataset(config["dataset"]["name"], split="validation")
    val_refs = [val[i][tgt_lang] for i in range(N)]
    val_src = [val[i][src_lang] for i in range(N)]

    ok1 = [r for _, r in beam1] == val_refs
    ok4 = [r for _, r in merged] == val_refs
    print(f"beam-1 REF lines match first {N} validation sentences in order: {'yes' if ok1 else 'no'}")
    print(f"merged REF lines match first {N} validation sentences in order: {'yes' if ok4 else 'no'}")
    if not (ok1 and ok4):
        fail("references do not match the validation set")

    hyp1 = [h for h, _ in beam1]
    hyp4 = [h for h, _ in merged]

    # ── 2. paired bootstrap ──
    print(f"\n=== 2. Paired bootstrap ({BOOTSTRAP_RUNS} resamples of {N} sentences, seed 42) ===")
    random.seed(42)
    scores1, scores4 = [], []
    for _ in range(BOOTSTRAP_RUNS):
        idx = random.choices(range(N), k=N)
        refs = [val_refs[i] for i in idx]
        scores1.append(sacrebleu.corpus_bleu([hyp1[i] for i in idx], [refs], force=True).score)
        scores4.append(sacrebleu.corpus_bleu([hyp4[i] for i in idx], [refs], force=True).score)
    s1, s4 = np.array(scores1), np.array(scores4)
    diff = s4 - s1
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"mean BLEU beam 1: {s1.mean():.2f}")
    print(f"mean BLEU beam 4: {s4.mean():.2f}")
    print(f"mean difference (beam 4 - beam 1): {diff.mean():.2f}")
    print(f"95% range of the difference: {lo:.2f} to {hi:.2f}")
    print(f"share of resamples where beam 4 scored higher: {100 * (s4 > s1).mean():.1f}%")

    # ── 3. where the outputs differ ──
    differ = [i for i in range(N) if hyp1[i] != hyp4[i]]
    print(f"\n=== 3. Outputs that differ between beam 1 and beam 4: {len(differ)}/{N} ===")
    shown = random.Random(42).sample(differ, min(10, len(differ)))
    print("(10 chosen at random from the differing ones, random.Random(42), shown in index order)")
    for i in sorted(shown):
        print(f"\n[{i}]\nGerman: {val_src[i]}\nbeam 1: {hyp1[i]}\nbeam 4: {hyp4[i]}\nREF:    {val_refs[i]}")

    # ── 4. worst beam-1 sentences ──
    print("\n=== 4. 15 beam-1 outputs with the lowest sentence-level BLEU ===")
    sent = [(sacrebleu.sentence_bleu(hyp1[i], [val_refs[i]]).score, i) for i in range(N)]
    sent.sort()
    for score, i in sent[:15]:
        print(f"\n[{i}] sentence BLEU {score:.1f}\nGerman: {val_src[i]}\nbeam 1: {hyp1[i]}\nREF:    {val_refs[i]}")
    print("\nAny reading of these 15 is my judgment, not a measurement.")

    # ── 5. no-eos ──
    print("\n=== 5. Ended without <eos> ===")
    print("For beam 4 this cannot be recomputed from the text files: whether a sequence "
          "ended with <eos> is not saved in them. No number is reported.")


if __name__ == "__main__":
    main()
