from pathlib import Path

import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

vocab_size = config["tokenizer"]["vocab_size"]
special_tokens = list(config["tokenizer"]["special_tokens"].values())
print(vocab_size, special_tokens)

ds = load_dataset(config["dataset"]["name"], split="train")
print(ds)

data_dir = ROOT / "data"
data_dir.mkdir(exist_ok=True)
corpus_path = data_dir / "combined_corpus.txt"

with open(corpus_path, "w", encoding="utf-8") as f:
    for example in ds:
        f.write(example["de"] + "\n")
        f.write(example["en"] + "\n")

print(f"Wrote {corpus_path}")

tokenizer = Tokenizer(BPE(unk_token=config["tokenizer"]["special_tokens"]["unk"]))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.train([str(corpus_path)], trainer)

print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

tokenizer_path = data_dir / "bpe_tokenizer.json"
tokenizer.save(str(tokenizer_path))
print(f"Saved tokenizer to {tokenizer_path}")

sample_de = ds[0]["de"]
sample_en = ds[0]["en"]

encoded_de = tokenizer.encode(sample_de)
encoded_en = tokenizer.encode(sample_en)

print(sample_de)
print(encoded_de.tokens)
print()
print(sample_en)
print(encoded_en.tokens)
