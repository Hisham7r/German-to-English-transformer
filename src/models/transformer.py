from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer

from decoder import Decoder, generate_causal_mask
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.decoder.token_embedding.embedding.weight = self.encoder.token_embedding.embedding.weight
        self.output_projection.weight = self.encoder.token_embedding.embedding.weight

    def generate_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1)  # [B, 1, S_src]

    def generate_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != self.pad_id).unsqueeze(1)  # [B, 1, S_tgt]
        causal_mask = generate_causal_mask(tgt_len).to(tgt.device)  # [1, S_tgt, S_tgt]
        return tgt_pad_mask * causal_mask  # broadcasts to [B, S_tgt, S_tgt]

    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)

        return logits


#------- Test-Block -------#

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab_size = config["tokenizer"]["vocab_size"]
    d_model = config["model"]["d_model"]
    num_heads = config["model"]["num_heads"]
    d_ff = config["model"]["d_ff"]
    num_layers = config["model"]["num_layers"]
    max_len = config["model"]["max_len"]
    dropout = config["model"]["dropout"]

    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    pad_id = tokenizer.token_to_id(config["tokenizer"]["special_tokens"]["pad"])

    src = torch.randint(0, vocab_size, (2, 10))
    tgt = torch.randint(0, vocab_size, (2, 8))

    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout, pad_id)
    logits = model(src, tgt)

    print("src shape:   ", src.shape)
    print("tgt shape:   ", tgt.shape)
    print("logits shape:", logits.shape)
    print()
    print("weight tying check:")
    print(
        "  encoder embedding IS decoder embedding:",
        model.encoder.token_embedding.embedding.weight is model.decoder.token_embedding.embedding.weight,
    )
    print(
        "  encoder embedding IS output_projection weight:",
        model.encoder.token_embedding.embedding.weight is model.output_projection.weight,
    )
