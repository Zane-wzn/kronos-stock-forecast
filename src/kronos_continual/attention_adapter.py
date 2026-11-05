"""Temporal residual pooling outside a frozen Kronos backbone.

A True padding mask entry excludes a key. The latest token is the query and
must be valid; all supplied keys are historical, so no triangular mask is used.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Last-query multihead attention, or a uniform-weight residual control."""

    def __init__(self, hidden_size=512, attention_width=32, num_heads=2, *, uniform=False):
        super().__init__()
        if min(hidden_size, attention_width, num_heads) < 1 or attention_width % num_heads:
            raise ValueError("positive dimensions and attention_width divisible by num_heads required")
        self.hidden_size = hidden_size
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.uniform = uniform
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, attention_width)
        self.k_proj = nn.Linear(hidden_size, attention_width)
        self.v_proj = nn.Linear(hidden_size, attention_width)
        self.out_proj = nn.Linear(attention_width, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if uniform:
            self.q_proj.requires_grad_(False)
            self.k_proj.requires_grad_(False)

    def forward(self, hidden_seq, key_padding_mask=None, *, return_weights=False):
        _validate_sequence(hidden_seq, self.hidden_size)
        valid = _valid_mask(hidden_seq, key_padding_mask)
        normalized = self.norm(hidden_seq)
        batch, time, _ = normalized.shape
        head_dim = self.attention_width // self.num_heads
        values = self.v_proj(normalized).view(batch, time, self.num_heads, head_dim).transpose(1, 2)
        if self.uniform:
            weights = valid[:, None, None, :].to(hidden_seq.dtype)
            weights = weights / weights.sum(-1, keepdim=True)
            weights = weights.expand(batch, self.num_heads, 1, time)
        else:
            query = self.q_proj(normalized[:, -1:]).view(batch, 1, self.num_heads, head_dim).transpose(1, 2)
            keys = self.k_proj(normalized).view(batch, time, self.num_heads, head_dim).transpose(1, 2)
            scores = (query @ keys.transpose(-2, -1)) / head_dim ** 0.5
            scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
            weights = scores.softmax(-1)
        attended = (weights @ values).transpose(1, 2).reshape(batch, self.attention_width)
        pooled = hidden_seq[:, -1] + self.out_proj(attended)
        if return_weights:
            return pooled, weights.mean(1)[:, 0]
        return pooled


class TemporalMeanPool(TemporalAttentionPool):
    """Uniform attention with the same normalization and value/output mappings.

    Retains the last-token residual and zero-output initialization of attention.
    Q/K are frozen; this control has fewer active parameters, not matched capacity.
    """

    def __init__(self, hidden_size=512, attention_width=32, num_heads=2):
        super().__init__(hidden_size, attention_width, num_heads, uniform=True)


class Adapter(nn.Module):
    def __init__(self, hidden_size=512, bottleneck=64, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        return hidden + self.up(self.drop(torch.nn.functional.gelu(self.down(self.ln(hidden)))))


class TemporalAttentionModel(nn.Module):
    def __init__(self, hidden_size=512, pred_len=10, bottleneck=64, dropout=0.1,
                 *, attention_width=32, num_heads=2, mode="temporal_attention"):
        super().__init__()
        if mode not in ("temporal_mean", "temporal_attention"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        pool = TemporalMeanPool if mode == "temporal_mean" else TemporalAttentionPool
        self.pool = pool(hidden_size, attention_width, num_heads)
        self.adapter = Adapter(hidden_size, bottleneck, dropout)
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, hidden_seq, key_padding_mask=None):
        return self.head(self.adapter(self.pool(hidden_seq, key_padding_mask)))


def _validate_sequence(x, hidden_size):
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[-1] != hidden_size:
        raise ValueError(f"expected nonempty [batch,time,{hidden_size}], got {tuple(x.shape)}")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("hidden_seq must contain finite floating point values")


def _valid_mask(x, mask):
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError("key_padding_mask must be bool [batch,time] on the input device")
    valid = ~mask
    if not valid[:, -1].all():
        raise ValueError("the last-token query must be valid")
    return valid
