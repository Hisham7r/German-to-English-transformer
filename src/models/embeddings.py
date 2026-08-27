import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

#------- Test-Block -------#
# and this block only runs if you execute this file directly, not when you import it as a module.

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab_size = config["tokenizer"]["vocab_size"]
    d_model = config["model"]["d_model"]
    max_len = config["model"]["max_len"]
    dropout = config["model"]["dropout"]

    token_embedding = TokenEmbedding(vocab_size, d_model)
    positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    fake_ids = torch.randint(0, vocab_size, (4, 10))  # batch=4, seq_len=10

    embedded = token_embedding(fake_ids)
    encoded = positional_encoding(embedded)

    print("input shape:", fake_ids.shape)
    print("token embedding shape:", embedded.shape)
    print("with positional encoding shape:", encoded.shape)
