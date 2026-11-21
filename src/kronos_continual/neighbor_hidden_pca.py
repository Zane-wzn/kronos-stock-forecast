"""Training-only neighbor selection and PCA on mean peer hidden vectors."""
from dataclasses import dataclass

import numpy as np
from torch import nn


def fit_neighbors(returns, k=16, overlap=120):
    clipped = returns.clip(returns.quantile(.01), returns.quantile(.99), axis=1)
    corr = clipped.corr(min_periods=overlap)
    return {
        stock: tuple(sorted(
            (other for other in corr.columns if other != stock and corr.at[stock, other] > 0),
            key=lambda other: (-corr.at[stock, other], other),
        )[:k])
        for stock in corr.columns
    }


def peer_mean(stocks, hidden, neighbors):
    hidden = np.asarray(hidden, dtype=np.float64)
    positions = {stock: pos for pos, stock in enumerate(stocks)}
    if hidden.ndim != 2 or len(positions) != len(hidden) or not np.isfinite(hidden).all():
        raise ValueError("Invalid daily stock panel")
    result = np.zeros_like(hidden)
    count = np.zeros(len(hidden), dtype=int)
    for row, stock in enumerate(stocks):
        ids = [positions[peer] for peer in neighbors.get(stock, ()) if peer in positions]
        count[row] = len(ids)
        if ids:
            result[row] = hidden[ids].mean(axis=0)
    return result, count


@dataclass(frozen=True)
class Context:
    mean: np.ndarray
    basis: np.ndarray
    eigen: np.ndarray
    scale: float
    rows: int
    variance_share: float

    def features(self, x, counts):
        centered = np.zeros_like(x, dtype=np.float64)
        keep = counts > 0
        centered[keep] = x[keep] - self.mean
        projected = centered @ self.basis
        weighted = projected * (self.eigen / self.eigen.sum())
        return projected, weighted / self.scale


def fit_context(days, rank=8):
    if rank < 1:
        raise ValueError("Rank must be positive")
    n = 0
    total = cross = None
    for x, counts in days:
        selected = np.asarray(x[counts > 0], dtype=np.float64)
        if not len(selected):
            continue
        if total is None:
            total = np.zeros(selected.shape[1])
            cross = np.zeros((len(total), len(total)))
        total += selected.sum(axis=0)
        cross += selected.T @ selected
        n += len(selected)
    if n <= rank:
        raise ValueError("Too few training rows with available peers")
    mean = total / n
    covariance = cross / n - np.outer(mean, mean)
    full, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    eigen, basis = full[::-1][:rank].copy(), vectors[:, ::-1][:, :rank].copy()
    if len(eigen) != rank or eigen[-1] <= max(eigen[0], 0) * 1e-12:
        raise ValueError("Insufficient nonzero principal directions")
    anchors = basis[np.argmax(np.abs(basis), axis=0), np.arange(rank)]
    basis *= np.where(anchors < 0, -1., 1.)
    energy = np.maximum(full, 0).sum()
    weights = eigen / eigen.sum()
    scale = float(np.sqrt(np.dot(eigen, weights ** 2) / eigen.sum()))
    return Context(mean, basis, eigen, scale, n, float(eigen.sum() / energy))


class AdditiveHead(nn.Module):
    def __init__(self, base, context_width):
        super().__init__()
        self.base = base
        self.neighbor = nn.Linear(context_width, base.out_features, bias=False)
        nn.init.zeros_(self.neighbor.weight)

    def forward(self, z, market):
        return self.base(z) + self.neighbor(market)


class NeighborForecast(nn.Module):
    def __init__(self, adapter, base, context_width=8):
        super().__init__()
        self.adapter = adapter
        self.head = AdditiveHead(base, context_width)
        self.widths = (base.in_features, context_width)

    def forward(self, joined):
        hidden, market = joined.split(self.widths, dim=-1)
        return self.head(self.adapter(hidden), market)
