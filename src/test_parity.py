from pathlib import Path

from datasets import load_dataset

from src.serve import ROOT, build_input_ids, load_model, special_ids, translate
from src.data.dataset import encode_source

N = 200
REFERENCE_FILE = ROOT / "experiments" / "validation_translations_beam1_start0_limit200_markers.txt"


def load_saved_hypotheses(path):
    # File format written by evaluate.py: "HYP: ...\nREF: ...\n\n" per sentence.
    text = Path(path).read_text(encoding="utf-8")
    return [line[len("HYP: "):] for line in text.split("\n") if line.startswith("HYP:")]


def main():
    model, tokenizer, config = load_model()
    _, sos_id, eos_id = special_ids(tokenizer, config)
    src_lang = config["dataset"]["source_lang"]

    validation = load_dataset(config["dataset"]["name"], split="validation")
    saved = load_saved_hypotheses(REFERENCE_FILE)
    assert len(saved) == N, f"expected {N} saved hypotheses, found {len(saved)}"

    matched = 0
    first_mismatch = None
    for i in range(N):
        german = validation[i][src_lang]

        reference_ids = encode_source(german, tokenizer, sos_id, eos_id)  # direct path
        serve_ids = build_input_ids(german, tokenizer, config)            # translate()'s path
        assert reference_ids == serve_ids, f"input ids differ at sentence {i}: {german!r}"

        got = translate(german, model, tokenizer, config)["translation"]
        expected = saved[i]
        if got == expected:
            matched += 1
        elif first_mismatch is None:
            first_mismatch = (i, german, expected, got)

    print(f"Matched {matched} out of {N}")
    if matched == N:
        print("PARITY TEST PASSED")
    else:
        i, german, expected, got = first_mismatch
        print(f"First mismatch at sentence {i}: {german!r}")
        print(f"  expected: {expected!r}")
        print(f"  got:      {got!r}")
        print("PARITY TEST FAILED")


if __name__ == "__main__":
    main()
