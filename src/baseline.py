from pathlib import Path

import sacrebleu
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ds = load_dataset(config["dataset"]["name"], split="test")

source_lang = config["dataset"]["source_lang"]
target_lang = config["dataset"]["target_lang"]

hypotheses = [example[source_lang] for example in ds]
references = [example[target_lang] for example in ds]

bleu = sacrebleu.corpus_bleu(hypotheses, [references])

print(f"Test set size: {len(ds)}")
print(f"Baseline (copy German unchanged) BLEU: {bleu.score:.2f}")
