import math

import torch


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, value)

    return output, attn_weights


#------- Test-Block -------#

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_k = 8

    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_k)

    output, attn_weights = scaled_dot_product_attention(query, key, value)

    print("query shape:      ", query.shape)
    print("output shape:     ", output.shape)
    print("attn_weights shape:", attn_weights.shape)
    print()
    print("attention weights for sentence 0, word 0:")
    print(attn_weights[0, 0])
    print("they sum to:", attn_weights[0, 0].sum().item())

    # now the same thing, but with the last 2 positions masked out as padding
    mask = torch.ones(batch_size, 1, seq_len)
    mask[:, :, 3:] = 0

    masked_output, masked_weights = scaled_dot_product_attention(query, key, value, mask)

    print()
    print("with last 2 positions masked as padding:")
    print(masked_weights[0, 0])
    print("they sum to:", masked_weights[0, 0].sum().item())
