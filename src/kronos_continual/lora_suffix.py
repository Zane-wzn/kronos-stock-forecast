"""Small LoRA building blocks; no dependency on PEFT or modification of vendor code."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=4, alpha=8):
        super().__init__()
        if rank < 1: raise ValueError('rank must be positive')
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


class KronosSuffix(nn.Module):
    """Recompute last blocks from a frozen prefix cache, using the original norm."""
    def __init__(self, blocks, norm, adapter_head, lora=False, rank=4, alpha=8):
        super().__init__()
        self.blocks = nn.ModuleList(blocks).requires_grad_(False)
        self.norm = norm.requires_grad_(False)
        self.adapter = adapter_head.adapter
        self.head = adapter_head.head
        if lora:
            for block in self.blocks:
                block.self_attn.q_proj = LoRALinear(block.self_attn.q_proj, rank, alpha)
                block.self_attn.v_proj = LoRALinear(block.self_attn.v_proj, rank, alpha)

    def forward(self, prefix):
        x = prefix
        for block in self.blocks: x = block(x)
        return self.head(self.adapter(self.norm(x)[:, -1]))


def ranking_loss(pred, target, score_scale):
    """Pairwise logistic loss WITHIN one origin date; ties ignored."""
    ps, ys = pred.mean(1), target.mean(1)
    i, j = torch.triu_indices(len(ps), len(ps), offset=1, device=ps.device)
    sign = torch.sign(ys[i] - ys[j])
    valid = sign != 0
    if not valid.any(): return pred.sum() * 0
    return F.softplus(-sign[valid] * (ps[i[valid]] - ps[j[valid]]) / score_scale).mean()


def objective(pred, target, scales, score_scale, rank_weight=0.):
    if rank_weight == 0:
        path = F.smooth_l1_loss(pred, target)
        return path, path.detach(), pred.detach().sum() * 0
    path = F.smooth_l1_loss(pred / scales, target / scales)
    rank = ranking_loss(pred, target, score_scale)
    return path + rank_weight * rank, path.detach(), rank.detach()
