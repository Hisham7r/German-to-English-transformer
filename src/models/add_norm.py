from pathlib import Path

import torch
import torch.nn as nn
import yaml


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        added = x + sublayer_output
        normalized = self.norm(added)
        return self.dropout(normalized)


#------- Test-Block -------#

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    d_model = config["model"]["d_model"]
    dropout = config["model"]["dropout"]

    batch_size = 2
    seq_len = 5

    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_output = torch.randn(batch_size, seq_len, d_model)

    add_norm = AddNorm(d_model, dropout)
    output = add_norm(x, sublayer_output)

    print("input shape: ", x.shape)
    print("output shape:", output.shape)
    print()
    print("mean of one word's vector (should be close to 0):", output[0, 0].mean().item())
    print("std of one word's vector (should be close to 1):", output[0, 0].std().item())
    