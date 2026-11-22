import argparse

import sacrebleu

from analyze_outputs import has_repeated_ngram, load_pairs, longest_run


def main():
    parser = argparse.ArgumentParser(description="Score one or more saved HYP/REF results files as one run.")
    parser.add_argument("files", nargs="+", help="results files, in sentence order (parts are joined)")
    args = parser.parse_args()

    hypotheses, references = [], []
    for path in args.files:
        h, r = load_pairs(path)
        print(f"{path}: {len(h)} sentences")
        hypotheses += h
        references += r

    n = len(hypotheses)
    hard_loops = sum(longest_run(h.split()) >= 3 for h in hypotheses)
    repeated_phrases = sum(has_repeated_ngram(h.split(), 3) for h in hypotheses)

    # Same lines evaluate.py prints, except "ended without <eos>" and timing:
    # those are not stored in the results file.
    print(f"\nSet size: {n}")
    print(sacrebleu.corpus_bleu(hypotheses, [references], force=True))
    print(f"hard loops (same word 3+ times in a row): {hard_loops} ({100 * hard_loops / n:.1f}%)")
    print(f"repeated 3-word phrase:                   {repeated_phrases} ({100 * repeated_phrases / n:.1f}%)")


if __name__ == "__main__":
    main()
