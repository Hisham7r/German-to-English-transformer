from pathlib import Path

import torch
import torch.nn as nn
import yaml


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


#------- Test-Block -------#

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    dropout = config["model"]["dropout"]

    batch_size = 2
    seq_len = 5

    x = torch.randn(batch_size, seq_len, d_model)

    ff = FeedForward(d_model, d_ff, dropout)
    output = ff(x)

    print("input shape: ", x.shape)
    print("output shape:", output.shape)
