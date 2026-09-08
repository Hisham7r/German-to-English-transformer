from pathlib import Path

import torch
import torch.nn as nn
import yaml

from add_norm import AddNorm
from embeddings import PositionalEncoding, TokenEmbedding
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        residual = x
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.add_norm1(residual, attn_output)

        residual = x
        ff_output = self.feed_forward(x)
        x = self.add_norm2(residual, ff_output)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


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

    batch_size = 2
    seq_len = 10

    fake_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    output = encoder(fake_ids)

    print("input shape: ", fake_ids.shape)
    print("output shape:", output.shape)
