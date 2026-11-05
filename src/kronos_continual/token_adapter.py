"""Adapter adaptation of Kronos' native coarse/fine token prediction path.

The tokenizer, backbone, dual token heads and dependency layer stay frozen.
Only the shared residual Adapter is trained. Vendor code is never modified.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .attention_adapter import Adapter


class TokenPredictionAdapter(nn.Module):
    """Adapt post-norm sequence context for both native token heads.

    New adapters start as identity. Reusing a return-trained Adapter is explicit:
    it changes the initial token distribution and need not preserve its quality.
    The old 512->10 return head is deliberately not part of this model.
    """

    def __init__(self, backbone: nn.Module, bottleneck: int = 64, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.d_model = backbone.d_model
        self.s1_vocab_size = backbone.s1_vocab_size
        self.adapter = Adapter(self.d_model, bottleneck, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        # eval disables frozen dropout; it does NOT disable Adapter autograd.
        self.backbone.eval()
        return self

    def load_return_adapter(self, checkpoint: str | Path):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("expected a plain return-model state_dict")
        adapter = {k.removeprefix("adapter."): v for k, v in state.items() if k.startswith("adapter.")}
        if not adapter:
            raise ValueError("checkpoint contains no adapter.* parameters")
        self.adapter.load_state_dict(adapter, strict=True)

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        # Recompute frozen contexts at every generated prefix. Static final-hidden
        # caches cannot stand in for contexts of generated future tokens.
        with torch.no_grad():
            _, context = self.backbone.decode_s1(s1_ids, s2_ids, stamp, padding_mask)
        adapted = self.adapter(context)
        return self.backbone.head(adapted), adapted

    def decode_s2(self, context, s1_ids, padding_mask=None):
        # Official autoregressive inference uses this exact native dependency path.
        return self.backbone.decode_s2(context, s1_ids, padding_mask)

    def _causal_fine_logits(self, context, next_s1, padding_mask=None, start=0):
        """Match native generation one query/prefix at a time.

        The vendor implementation's single-query RoPE broadcasts across keys.
        A parallel equal-length Q/K call has DIFFERENT position semantics, even
        with a causal mask. Prefix calls preserve native decode_s2 exactly and
        prevent future access without toggling frozen layers to train mode.
        Historical positions before start are unused zeros for masked CE only.
        """
        if next_s1.shape != context.shape[:2] or not 0 <= start < context.shape[1]:
            raise ValueError("coarse IDs and fine start must match context")
        pieces = []
        if start:
            vocab = self.backbone.head.vocab_s2
            pieces.append(context.new_zeros(context.shape[0], start, vocab))
        for i in range(start, context.shape[1]):
            mask = None if padding_mask is None else padding_mask[:, :i+1]
            logits = self.decode_s2(context[:, :i+1], next_s1[:, i:i+1], mask)
            pieces.append(logits[:, -1:])
        return torch.cat(pieces, dim=1)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, *,
                use_teacher_forcing=False, s1_targets=None, fine_start=0):
        """Shifted input tokens; fine conditions on sampled predicted coarse IDs.

        Teacher forcing coarse IDs is available for diagnostics only. It is never
        used by the experiment default or generation interfaces.
        """
        logits1, context = self.decode_s1(s1_ids, s2_ids, stamp, padding_mask)
        if use_teacher_forcing:
            if s1_targets is None or s1_targets.shape != s1_ids.shape:
                raise ValueError("teacher forcing requires matching shifted coarse targets")
            coarse = s1_targets
        else:
            prob = logits1.detach().softmax(-1)
            coarse = torch.multinomial(prob.reshape(-1, self.s1_vocab_size), 1).reshape(s1_ids.shape)
        return logits1, self._causal_fine_logits(context, coarse, padding_mask, start=fine_start)


def shifted_future_loss(logits1, logits2, token1, token2, history_length):
    """Mask historical next-token targets; supervise exactly the future suffix.

    For 90 context + 10 target tokens, input is tokens[0:99], targets[1:100],
    and supervised input positions are 89..98 (targets 90..99).
    """
    if token1.shape != token2.shape or token1.ndim != 2:
        raise ValueError("coarse/fine tokens must share [batch,total_length]")
    if not 1 <= history_length < token1.shape[1]:
        raise ValueError("need nonempty history and future")
    expected = (*token1.shape[:1], token1.shape[1] - 1)
    if logits1.shape[:2] != expected or logits2.shape[:2] != expected:
        raise ValueError("logits must align with tokens[:, :-1]")
    start = history_length - 1
    loss1 = F.cross_entropy(logits1[:, start:].reshape(-1, logits1.shape[-1]),
                           token1[:, history_length:].reshape(-1))
    loss2 = F.cross_entropy(logits2[:, start:].reshape(-1, logits2.shape[-1]),
                           token2[:, history_length:].reshape(-1))
    return (loss1 + loss2) / 2, loss1, loss2


def normalize_history_future(history, future, clip=5.0):
    """Use HISTORY statistics only, including for future training targets."""
    import numpy as np
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if history.ndim != 2 or future.ndim != 2 or history.shape[1] != future.shape[1]:
        raise ValueError("expected compatible [time,fields] arrays")
    if not len(history) or not len(future) or not np.isfinite(history).all() or not np.isfinite(future).all():
        raise ValueError("finite nonempty history/future required")
    mean, scale = history.mean(0), history.std(0) + 1e-5
    full = np.concatenate([history, future], axis=0)
    return np.clip((full - mean) / scale, -clip, clip).astype(np.float32), mean, scale
