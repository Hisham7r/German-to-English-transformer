import re
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "test_translations.txt"


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    hypotheses = []
    references = []

    for entry in content.strip().split("\n\n"):
        lines = entry.strip().split("\n")
        if len(lines) < 2:
            continue
        hyp_line, ref_line = lines[0], lines[1]
        hypotheses.append(hyp_line.removeprefix("HYP: "))
        references.append(ref_line.removeprefix("REF: "))

    return hypotheses, references


def score(hypotheses, references):
    return sacrebleu.corpus_bleu(hypotheses, [references], force=True).score


def fix_punctuation_spacing(text):
    # the detokenize.py heuristic treats punctuation as its own "word", so it
    # gets a space before it (e.g. "something ."); strip that space back out
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def longest_run(words):
    if not words:
        return 0
    longest = current = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def has_repeated_ngram(words, n):
    if len(words) < n:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def repetition_report(sentences, name, length_threshold):
    total = len(sentences)
    long_run_count = 0
    repeated_ngram_count = 0
    overlong_count = 0

    for sentence in sentences:
        words = sentence.split()
        if longest_run(words) >= 3:
            long_run_count += 1
        if has_repeated_ngram(words, 3):
            repeated_ngram_count += 1
        if len(words) > length_threshold:
            overlong_count += 1

    print(f"\n{name} (n={total}):")
    print(f"  longest run >= 3 same word:              {long_run_count} ({100 * long_run_count / total:.1f}%)")
    print(f"  repeated 3-word sequence:                 {repeated_ngram_count} ({100 * repeated_ngram_count / total:.1f}%)")
    print(f"  longer than any reference ({length_threshold} words):     {overlong_count} ({100 * overlong_count / total:.1f}%)")


if __name__ == "__main__":
    hypotheses, references = load_pairs(RESULTS_PATH)
    print(len(hypotheses), len(references))

    print("BLEU as saved:", score(hypotheses, references))   # expect 24.34

    fixed = [fix_punctuation_spacing(h) for h in hypotheses]
    print("BLEU with punctuation fix:", score(fixed, references))
    
    changed = sum(f != h for f, h in zip(fixed, hypotheses))
    print("sentences changed by the fix:", changed)

    threshold = max(len(r.split()) for r in references)

    # What-if diagnostic, NOT a real fix: cut every output to the first
    # `threshold` words (the longest reference's length) and re-score. It shows
    # how much BLEU is lost to over-long, repetitive tails. Real improvement has
    # to come from decoding (beam search / repetition control), not from cutting.
    truncated = [" ".join(h.split()[:threshold]) for h in hypotheses]
    cut_changed = sum(t != h for t, h in zip(truncated, hypotheses))
    print(f"outputs actually changed by truncating to {threshold} words:", cut_changed)

    print("BLEU summary, one under the other:")
    print("  original outputs: ", sacrebleu.corpus_bleu(hypotheses, [references], force=True))
    print("  truncated outputs:", sacrebleu.corpus_bleu(truncated, [references], force=True))

    repetition_report(hypotheses, "model outputs", threshold)
    repetition_report(references, "references (control)", threshold)
