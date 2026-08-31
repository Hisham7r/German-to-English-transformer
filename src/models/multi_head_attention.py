from pathlib import Path

import torch
import torch.nn as nn
import yaml

from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch, q_len, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch, k_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch, k_len, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn_weights = scaled_dot_product_attention(query, key, value, mask)

        output = output.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        output = self.output_linear(output)

        return output, attn_weights


#------- Test-Block -------#

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    d_model = config["model"]["d_model"]
    num_heads = config["model"]["num_heads"]

    batch_size = 2
    seq_len = 5

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(query, key, value)

    print("query shape:       ", query.shape)
    print("output shape:      ", output.shape)
    print("attn_weights shape:", attn_weights.shape)
    print()
    print("attention weights for sentence 0, head 0, word 0:")
    print(attn_weights[0, 0, 0])
    print("they sum to:", attn_weights[0, 0, 0].sum().item())

    # same thing, but with the last 2 positions masked out as padding
    mask = torch.ones(batch_size, 1, seq_len)
    mask[:, :, 3:] = 0

    masked_output, masked_weights = mha(query, key, value, mask)

    print()
    print("with last 2 positions masked as padding:")
    print("masked_output shape:", masked_output.shape)
    print("weights for sentence 0, head 0, word 0:")
    print(masked_weights[0, 0, 0])
    print("they sum to:", masked_weights[0, 0, 0].sum().item())
