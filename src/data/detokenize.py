"""
Best-effort rejoining of generated subword ids into readable text.

The BPE tokenizer (src/data/tokenizer.py) was trained with a Whitespace
pre-tokenizer and no continuing-subword-prefix / end-of-word-suffix, so
individual vocab entries carry no marker for "I start a new word" vs.
"I glue onto the piece before me" (PROGRESS.md Known Issues #2). That
distinction exists only at encode time, via Encoding.word_ids -- it is
not part of the vocab, so it is not available for ids the model itself
generates one at a time during greedy/beam decoding.

This approximates it: for every token id, look at how it behaved across
the training corpus (did it usually start a pre-tokenizer word, or
continue one?) and use that as a per-id rule at decode time. It is a
heuristic, not an exact inverse of encoding -- an id that goes both ways
in training gets whichever behavior was more common, so some sentences
will still join a little wrong. The real fix is retraining the tokenizer
with a boundary marker (e.g. continuing_subword_prefix) and retraining
the model against the new vocab; that would invalidate the current
trained checkpoint, so it stays separate follow-up work, not done here.
"""

from collections import Counter


def build_boundary_map(tokenizer, sentences):
    """
    sentences: iterable of raw strings (train split only -- same
    no-leakage rule as tokenizer training).

    Returns {token_id: bool}, True meaning "this id usually starts a new
    word, put a space before it when rejoining".
    """
    starts = Counter()
    continues = Counter()

    for sentence in sentences:
        encoding = tokenizer.encode(sentence)
        prev_word_id = None
        for token_id, word_id in zip(encoding.ids, encoding.word_ids):
            if word_id != prev_word_id:
                starts[token_id] += 1
            else:
                continues[token_id] += 1
            prev_word_id = word_id

    all_ids = set(starts) | set(continues)
    return {token_id: starts[token_id] >= continues[token_id] for token_id in all_ids}


def clean_decode(ids, tokenizer, boundary_map, special_ids):
    """
    Turn a list of generated token ids into a readable string: skip
    special tokens (pad/sos/eos) and use boundary_map to decide spacing.
    An id never seen while building the map defaults to "starts a new
    word" -- safer than silently gluing unknown pieces together.
    """
    pieces = []
    for token_id in ids:
        if token_id in special_ids:
            continue
        token_str = tokenizer.id_to_token(token_id)
        starts_word = boundary_map.get(token_id, True)
        if pieces and not starts_word:
            pieces[-1] += token_str
        else:
            pieces.append(token_str)
    return " ".join(pieces)


# ── Test block ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    from tokenizers import Tokenizer

    ROOT = Path(__file__).resolve().parents[2]
    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))

    # A handful of sentences instead of the full ~58k-sentence train corpus --
    # this only checks the mechanics, not real coverage.
    sample_sentences = [
        "Zwei junge weisse Maenner sind im Freien in der Naehe vieler Buesche.",
        "Ein Mann in einem blauen Hemd spielt Gitarre.",
        "Mehrere Maenner spielen Fussball auf einem Feld.",
    ]
    boundary_map = build_boundary_map(tokenizer, sample_sentences)

    test_sentence = sample_sentences[0]
    ids = tokenizer.encode(test_sentence).ids

    print("Original:       ", test_sentence)
    print("Tokens:          ", tokenizer.encode(test_sentence).tokens)
    print("Default decode:  ", tokenizer.decode(ids))
    print("Clean decode:    ", clean_decode(ids, tokenizer, boundary_map, special_ids=set()))
