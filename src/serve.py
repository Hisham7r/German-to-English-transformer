import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# src/decode.py imports its sibling package by the bare name `models`
# (`from models.decoder import ...`), which only resolves when src/ itself is on
# sys.path. Under `python -m src.serve` only the repo root is, so add src/ here.
sys.path.insert(0, str(ROOT / "src"))

from src.data.dataset import encode_source
from src.data.detokenize import build_boundary_map, clean_decode
from src.decode import greedy_decode
from src.models.transformer import Transformer

DEFAULT_WEIGHTS = "experiments/best_model_weights.pt"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
TOKENIZER_PATH = ROOT / "data" / "bpe_tokenizer.json"

DEVICE = torch.device("cpu")


def load_model(weights_path=DEFAULT_WEIGHTS):
    """Returns (model, tokenizer, config). Everything translate() needs is in
    those three: the boundary map that clean_decode requires is built here once
    (slow: it tokenizes the whole train split) and stored in config["boundary_map"].
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    special = config["tokenizer"]["special_tokens"]
    pad_id = tokenizer.token_to_id(special["pad"])

    model = Transformer(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"],
        pad_id=pad_id,
    )

    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path
    saved = torch.load(weights_path, map_location="cpu")
    # train.py saves a checkpoint dict; accept that or a bare state dict.
    state_dict = saved["model_state"] if "model_state" in saved else saved
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    train_raw = load_dataset(config["dataset"]["name"], split="train")
    train_sentences = list(train_raw[config["dataset"]["source_lang"]]) + list(
        train_raw[config["dataset"]["target_lang"]]
    )
    config["boundary_map"] = build_boundary_map(tokenizer, train_sentences)

    return model, tokenizer, config


def special_ids(tokenizer, config):
    special = config["tokenizer"]["special_tokens"]
    return (
        tokenizer.token_to_id(special["pad"]),
        tokenizer.token_to_id(special["sos"]),
        tokenizer.token_to_id(special["eos"]),
    )


def build_input_ids(text, tokenizer, config):
    """Step 1 of translate(), kept separate so the parity test can check it."""
    _, sos_id, eos_id = special_ids(tokenizer, config)
    return encode_source(text, tokenizer, sos_id, eos_id)


def translate(text: str, model, tokenizer, config) -> dict:
    pad_id, sos_id, eos_id = special_ids(tokenizer, config)

    start = time.time()

    # 1. German text -> ids, with <sos> ... <eos>
    src_ids = build_input_ids(text, tokenizer, config)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_mask = (src != pad_id).unsqueeze(1)

    # 2. Greedy decode (greedy_decode already runs under torch.no_grad())
    with torch.no_grad():
        output_ids = greedy_decode(
            model, src, src_mask,
            max_len=config["model"]["max_len"],
            sos_id=sos_id, eos_id=eos_id, device=DEVICE,
        )

    # 3. ids -> readable English
    english = clean_decode(
        output_ids, tokenizer, config["boundary_map"], {pad_id, sos_id, eos_id}
    )

    elapsed = time.time() - start
    return {"translation": english, "time_taken": round(elapsed, 3)}


if __name__ == "__main__":
    model, tokenizer, config = load_model()
    result = translate("Ein Mann geht durch die Stadt.", model, tokenizer, config)
    print(result)
