"""Construct causal all-stock PCA corrections through the current Adapter head."""
import numpy as np
import pandas as pd
import torch

from scripts.experiments import peer_pca_replay as peer
from src.kronos_continual.neighbor_hidden_pca import fit_context
from src.kronos_continual.online_peer_pca_head import OnlinePCA, cross_section_weights, weighted_feature_mean

MODE = "all_peers_online_weighted"
WINDOW = 120
MIN_OVERLAP = 40
HALF_LIFE = 30


def neighbor_protocol(cfg, panels, window):
    if window < MIN_OVERLAP:
        raise ValueError(f"Window must permit at least {MIN_OVERLAP} overlapping returns")
    closes = []
    for split, meta in panels.items():
        values = np.load(cfg["cache"] / f"current_{split}.npy", mmap_mode="r")
        closes.append(meta[peer.KEYS].assign(close=np.where(values > 0, values, np.nan)))
    prices = pd.concat(closes).pivot(index="as_of_date", columns="stock", values="close").sort_index()
    returns = prices.pct_change(fill_method=None)
    # The retained panel skips dates at the split boundaries.
    boundaries = [meta.as_of_date.min() for split, meta in panels.items() if split != "train"]
    returns.loc[boundaries] = np.nan
    train_start = panels["train"].as_of_date.min()
    train_end = panels["train"].as_of_date.max()
    bounds = returns.loc[train_start:train_end].quantile([.01, .99])
    returns = returns.clip(lower=bounds.loc[.01], upper=bounds.loc[.99], axis=1)
    stocks = {stock: i for i, stock in enumerate(prices.columns)}
    values = returns.to_numpy()
    days = returns.index
    weights = {}
    for date in sorted(set(panels["val"].as_of_date) | set(panels["test"].as_of_date)):
        end = days.get_loc(date)
        weights[date] = cross_section_weights(values[max(0, end - window):end], MIN_OVERLAP)
    return stocks, weights


def fit_training_context(train, hidden, encoder):
    """Fit PCA on the offline Adapter's leave-one-out training means."""
    encoded = np.empty(hidden.shape, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(hidden), 8192):
            batch = slice(start, start + 8192)
            encoded[batch] = encoder(torch.tensor(hidden[batch], dtype=torch.float32)).numpy()

    def training_samples():
        for ids in train.groupby("as_of_date", sort=True).indices.values():
            features = encoded[ids].astype(float)
            n = len(ids) - 1
            yield (features.sum(axis=0) - features) / max(n, 1), np.full(len(ids), n)

    return fit_context(training_samples())


def correction_observer(meta, stocks, weights, context):
    stock_ids = meta.stock.map(stocks).to_numpy(dtype=int)
    pca = OnlinePCA(context.mean, context.basis, context.eigen, decay=2 ** (-1 / HALF_LIFE))
    corrections = np.empty((len(meta), 10), dtype=float)
    counts = np.empty(len(meta), dtype=int)
    updated_dates = []

    def observe(date, ids, encoded, head):
        assert isinstance(head, torch.nn.Linear) and (head.in_features, head.out_features) == (512, 10)
        features = encoded.cpu().numpy().astype(float)
        head_weight = head.weight.detach().cpu().numpy().astype(float)
        day_stocks = stock_ids[ids]
        peers = ~np.eye(len(ids), dtype=bool)
        points, available_counts = weighted_feature_mean(
            features, peers, weights[date][np.ix_(day_stocks, day_stocks)])
        counts[ids] = available_counts
        available = available_counts > 0
        corrections[ids] = pca.reconstruct(points, available) @ head_weight.T
        if available.any():
            pca.update(points, available)
            updated_dates.append(date)

    return observe, corrections, counts, updated_dates


def verify_corrections(corrections, counts):
    assert np.isfinite(corrections).all() and not np.any(corrections[counts == 0])
