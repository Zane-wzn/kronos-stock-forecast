"""Portable HCA inference from frozen Kronos last-token hidden features."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, hidden_dim=512, bottleneck=64):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.drop = nn.Dropout(0.1)
        self.up = nn.Linear(bottleneck, hidden_dim)

    def forward(self, hidden):
        return hidden + self.up(self.drop(F.gelu(self.down(self.ln(hidden)))))


class ResidualHead(nn.Module):
    def __init__(self, hidden_dim=512, horizon=10, rank=2):
        super().__init__()
        self.base = nn.Linear(hidden_dim, horizon)
        self.coeff = nn.Linear(hidden_dim, rank)
        self.register_buffer("basis", torch.zeros(horizon, rank))

    def forward(self, z):
        return self.base(z) + self.coeff(z) @ self.basis.T


class PortableHCA(nn.Module):
    def __init__(self, residual=True):
        super().__init__()
        self.adapter = Adapter()
        self.head = ResidualHead() if residual else nn.Linear(512, 10)

    def forward(self, hidden):
        return self.head(self.adapter(hidden))


def load_model(path: str | Path, device="cpu"):
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    model = PortableHCA(residual="head.basis" in state)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
