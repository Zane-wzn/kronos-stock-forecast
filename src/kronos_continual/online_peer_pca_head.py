"""Trailing neighbor weights and exponentially updated peer PCA."""

import numpy as np
from scipy.linalg import eigh


def correlation_weights(history, neighbors, minimum=40):
    """Positive Pearson correlation on overlapping return dates."""
    valid = neighbors >= 0
    own = np.broadcast_to(history.T[:, None, :], (*neighbors.shape, len(history)))
    peers = history.T[np.maximum(neighbors, 0)]
    shared = np.isfinite(own) & np.isfinite(peers)
    counts = shared.sum(axis=-1)
    own = np.where(shared, own, 0.)
    peers = np.where(shared, peers, 0.)
    denominator = np.maximum(counts, 1)
    own = np.where(shared, own - own.sum(axis=-1, keepdims=True) / denominator[..., None], 0.)
    peers = np.where(shared, peers - peers.sum(axis=-1, keepdims=True) / denominator[..., None], 0.)
    scale = np.sqrt(np.einsum("ijk,ijk->ij", own, own) * np.einsum("ijk,ijk->ij", peers, peers))
    return np.divide(np.einsum("ijk,ijk->ij", own, peers), scale, out=np.zeros_like(scale),
                     where=valid & (counts >= minimum) & (scale > 0)).clip(0., 1.)


def aggregate_neighbors(hidden, stock_ids, neighbors, weights=None):
    """Combine available neighbors, falling back to equal weights if needed."""
    positions = np.full(len(neighbors), -1, dtype=int)
    positions[stock_ids] = np.arange(len(stock_ids))
    candidates = neighbors[stock_ids]
    indices = positions[np.maximum(candidates, 0)]
    available = (candidates >= 0) & (indices >= 0)
    counts = available.sum(axis=1)
    mass = normalized_weights(available, weights)
    x = np.einsum("ij,ijk->ik", mass, hidden[np.maximum(indices, 0)], optimize=True)
    return x, counts


def normalized_weights(available, weights):
    mass = available.astype(float)
    if weights is not None:
        mass *= weights
        mass = np.where(mass.sum(axis=1, keepdims=True) > 0, mass, available)
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    return mass


def weighted_feature_mean(hidden, available, weights=None):
    return normalized_weights(available, weights) @ hidden, available.sum(axis=1)


def cross_section_weights(history, minimum=40):
    """Pairwise Pearson weights with missing dates omitted for each pair."""
    observed = np.isfinite(history).astype(float)
    values = np.where(observed > 0, history, 0.)
    counts = observed.T @ observed
    total = values.T @ observed
    denominator = np.maximum(counts, 1.)
    covariance = values.T @ values - total * total.T / denominator
    variance = np.maximum((values ** 2).T @ observed - total ** 2 / denominator, 0.)
    scale = np.sqrt(variance * variance.T)
    weights = np.divide(covariance, scale, out=np.zeros_like(scale),
                        where=(counts >= minimum) & (scale > 0)).clip(0., 1.)
    np.fill_diagonal(weights, 0.)
    return weights


class OnlinePCA:
    def __init__(self, mean, basis, eigen, decay):
        self.mean = mean.copy()
        self.basis = basis.copy()
        self.covariance = (basis * eigen) @ basis.T
        self.decay = decay

    def reconstruct(self, points, available):
        centered = np.where(available[:, None], points - self.mean, 0.)
        return (centered @ self.basis) @ self.basis.T

    def update(self, points, available):
        points = points[available]
        if not len(points):
            return
        mean = points.mean(axis=0)
        centered = points - mean
        shift = mean - self.mean
        decay = self.decay
        self.covariance = (
            decay * self.covariance
            + (1 - decay) * (centered.T @ centered / len(points) + decay * np.outer(shift, shift))
        )
        self.mean += (1 - decay) * shift
        dimension, rank = self.basis.shape
        _, self.basis = eigh(self.covariance, subset_by_index=(dimension - rank, dimension - 1))
