import random
import re
import sys
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

FILES = {
    "beam 1": EXP / "validation_translations_beam1_start0_limit200_markers.txt",
    "beam 4": EXP / "validation_translations_beam4_start0_limit200_markers_joined.txt",
}

# Fix A: " - " between two letters -> "-"
HYPHEN = re.compile(r"(?<=[^\W\d_]) - (?=[^\W\d_])")
# Fix B: space before 's / n't / 're / 've / 'll / 'd / 'm removed
APOSTROPHE = re.compile(r"(?<=\w) (?=(?:'(?:s|re|ve|ll|d|m)|n't)\b)")


def fix_hyphen(text):
    return HYPHEN.sub("-", text)


def fix_apostrophe(text):
    return APOSTROPHE.sub("", text)


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


def has_hyphen(s):
    return " - " in s


def has_apostrophe(s):
    return " 's" in s or " n't" in s or " '" in s


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    data = {}
    for name, path in FILES.items():
        hyps, refs = load_entries(path)
        if len(hyps) != 200 or len(refs) != 200:
            sys.exit(f"STOP: {path.name} has {len(hyps)} HYP / {len(refs)} REF lines, expected 200")
        data[name] = (hyps, refs)
    if data["beam 1"][1] != data["beam 4"][1]:
        sys.exit("STOP: the REF lines of the two files differ")
    refs = data["beam 4"][1]

    print("=== 1. How many of the 200 contain the pattern ===")
    hyphen_header = '" - "'
    apostrophe_header = "space before apostrophe (' s, n't, ')"
    print(f"{'':<12}{hyphen_header:>8}{apostrophe_header:>46}")
    for name, (hyps, _) in data.items():
        print(f"{name:<12}{sum(map(has_hyphen, hyps)):>8}{sum(map(has_apostrophe, hyps)):>46}")
    print(f"{'references':<12}{sum(map(has_hyphen, refs)):>8}{sum(map(has_apostrophe, refs)):>46}")

    fixes = {
        "A. hyphen fix": fix_hyphen,
        "B. apostrophe fix": fix_apostrophe,
        "C. A and B together": lambda s: fix_apostrophe(fix_hyphen(s)),
    }

    print("\n=== 2. BLEU unfixed and after each fix ===")
    fixed_by_name = {}
    for name, (hyps, _) in data.items():
        base = sacrebleu.corpus_bleu(hyps, [refs], force=True)
        print(f"\n{name}\n  unfixed:             {base}")
        for fix_name, fn in fixes.items():
            fixed = [fn(h) for h in hyps]
            score = sacrebleu.corpus_bleu(fixed, [refs], force=True)
            print(f"  {fix_name + ':':<20} {score}\n  {'':<20} difference from unfixed: {score.score - base.score:+.2f}")
            if fix_name.startswith("C"):
                fixed_by_name[name] = fixed

    print("\n=== 3. 8 beam-4 sentences where the fixes (C) changed the text ===")
    hyps4 = data["beam 4"][0]
    fixed4 = fixed_by_name["beam 4"]
    changed = [i for i in range(200) if hyps4[i] != fixed4[i]]
    print(f"{len(changed)}/200 beam-4 sentences changed; 8 chosen at random (random.Random(42)), shown in index order")
    for i in sorted(random.Random(42).sample(changed, min(8, len(changed)))):
        print(f"\n[{i}]\nbefore: {hyps4[i]}\nafter:  {fixed4[i]}\nREF:    {refs[i]}")

    print("\n=== 4. Glued words ===")
    print("Glued words such as 'photographedor' or 'yellowletters' cannot be repaired from the saved text: "
          "the text no longer shows where the boundary should be. Nothing here tries to.")


if __name__ == "__main__":
    main()
