from pathlib import Path

import torch
import torch.nn as nn
import yaml

from add_norm import AddNorm
from embeddings import PositionalEncoding, TokenEmbedding
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention


def generate_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        residual = x
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.add_norm1(residual, self_attn_output)

        residual = x
        cross_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.add_norm2(residual, cross_output)

        residual = x
        ff_output = self.feed_forward(x)
        x = self.add_norm3(residual, ff_output)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
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
    tgt_seq_len = 8
    src_seq_len = 10

    fake_tgt_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    fake_encoder_output = torch.randn(batch_size, src_seq_len, d_model)

    tgt_mask = generate_causal_mask(tgt_seq_len)

    decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    output = decoder(fake_tgt_ids, fake_encoder_output, tgt_mask=tgt_mask)

    print("fake_tgt_ids shape:       ", fake_tgt_ids.shape)
    print("fake_encoder_output shape:", fake_encoder_output.shape)
    print("output shape:             ", output.shape)
    print()
    print("causal mask (1 = can see, 0 = blocked):")
    print(tgt_mask.squeeze(0))
