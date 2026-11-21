"""Replay the CSI300 all-stock rolling-weighted PCA with a 60-day window."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from evaluate.metrics import _direction_metrics
from scripts.evaluation import peer_pca_metrics as metrics
from scripts.experiments import peer_pca_correction as pca
from scripts.experiments import peer_pca_replay as peer
MARKET = "csi300"
PRIOR = peer.CONTROLS / MARKET
OUT = peer.RESULTS.joinpath("csi300_all_peer_pca_60d_20261121")
MODE = pca.MODE
CONTROL = "rolling_120d"
VARIANT = "rolling_60d"
WINDOW = 60


def run(out):
    cfg = peer.config(MARKET)
    cache = cfg["cache"]
    train = pd.read_csv(cache / "meta_train.csv", dtype=str, usecols=peer.KEYS)
    h_train = np.load(cache / "hidden_train.npy", mmap_mode="r")
    val = pd.read_csv(cache / "meta_val.csv", dtype=str, usecols=[*peer.KEYS, "label_end_date"])
    test, h_test = peer.panel(cache, "test")
    labels = np.load(cache / "labels_test.npy", mmap_mode="r")
    assert h_train.shape == (len(train), 512)
    assert train.as_of_date.max() < val.as_of_date.min() <= val.label_end_date.max() < test.as_of_date.min()
    dates = sorted(test.as_of_date.unique())
    assert (len(dates), len(test)) == peer.EXPECTED[MARKET][:2]
    alpha = peer.read_json(PRIOR / "selected_alpha.json")["selected"][MODE]["alpha"]
    stocks, weights = pca.neighbor_protocol(
        cfg, dict(train=train, val=val, test=test), WINDOW)
    old_metrics = pd.read_csv(peer.CONTROLS / "metrics_by_seed.csv").query("market == @MARKET")
    metric_parts, daily, audits = [], [], []
    day_rows = test.groupby("as_of_date", sort=True).indices
    stock_ids = test.stock.map(stocks).to_numpy(dtype=int)
    fallback = 0
    for date, ids in day_rows.items():
        current = stock_ids[ids]
        fallback += int((weights[date][np.ix_(current, current)].sum(axis=1) == 0).sum())

    for seed in peer.SEEDS:
        print(f"{MARKET} seed{seed}: fit train-only PCA and replay", flush=True)
        seed_out = out / f"seed{seed}"
        seed_out.mkdir(exist_ok=True)
        source = cfg["home"] / "predictions" / f"Prediction_online_seed{seed}.csv.gz"
        archive = pd.read_csv(source, usecols=peer.KEYS + peer.PC + peer.AC,
                              dtype={"stock": str}, float_precision="round_trip")
        base = peer.align_predictions(test, archive)
        actual = peer.align_predictions(test, archive, peer.AC)
        np.testing.assert_allclose(actual, labels, atol=5e-7, rtol=0)
        with np.load(PRIOR / f"seed{seed}" / "predictions.npz") as previous:
            np.testing.assert_array_equal(previous["A"], base)
            control = previous[MODE]
        schedule, encoder, audit = peer.supervision(MARKET, cfg, test, seed)
        contexts = pca.fit_training_context(train, h_train, encoder)
        observe, corrections, counts, updates = pca.correction_observer(test, stocks, weights, contexts)
        _, replay, versions = peer.baseline_embeddings(
            MARKET, cfg, Path(audit["checkpoint"]), encoder, test, h_test, labels,
            base, schedule, "cpu", observe=observe)
        expected = np.cumsum([0, *(date in schedule for date in dates)])
        np.testing.assert_array_equal(versions, expected[:-1])
        assert expected[-1] == peer.EXPECTED[MARKET][2] and len(updates) == len(dates)
        assert replay["baseline_prediction_max_error"] == 0.
        pca.verify_corrections(corrections, counts)
        predictions = {"A": base, CONTROL: control, VARIANT: base + alpha * corrections}
        np.savez_compressed(seed_out / "predictions.npz", **predictions)
        metrics.evaluate(cfg, test, actual, predictions, seed, seed_out)
        result = pd.read_csv(seed_out / "metrics.csv")
        for current, old in (("A", "A"), (CONTROL, MODE)):
            cols = ["rankic", "net_cumulative_return", "sharpe", "max_drawdown"]
            saved = old_metrics[(old_metrics.seed == seed) & (old_metrics.model == old)].set_index("signal")[cols]
            recomputed = result[result.model == current].set_index("signal")[cols]
            pd.testing.assert_frame_equal(saved.sort_index(), recomputed.sort_index(), atol=1e-12, rtol=0)
        for model, pred in predictions.items():
            for signal, forecast, target in (
                ("endpoint", pred[:, -1], actual[:, -1]),
                ("path", pred.mean(axis=1), actual.mean(axis=1)),
            ):
                direction = _direction_metrics(forecast, target)
                mask = (result.model == model) & (result.signal == signal)
                result.loc[mask, "direction_accuracy"] = direction["accuracy"]
                result.loc[mask, "balanced_direction_accuracy"] = direction["balanced_accuracy"]
                result.loc[mask, "signal_rmse"] = np.sqrt(np.mean((forecast - target) ** 2))
        metric_parts.append(result)
        daily.append(pd.read_csv(seed_out / "daily_metrics.csv"))
        audits.append(dict(seed=seed, pca_updates=len(updates), baseline_updates=int(expected[-1]), **audit, **replay))

    by_seed = pd.concat(metric_parts, ignore_index=True)
    assert len(by_seed) == 18 and not by_seed.duplicated(["model", "signal", "seed"]).any()
    by_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    cols = ["net_cumulative_return", "rankic", "sharpe", "max_drawdown", "direction_accuracy",
            "balanced_direction_accuracy", "signal_rmse", "turnover"]
    summary = by_seed.groupby(["signal", "model"])[cols].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(out / "summary_mean_std.csv")
    metrics.paired(pd.concat(daily, ignore_index=True), VARIANT, ("A", CONTROL)).to_csv(
        out / "paired_intervals.csv", index=False)
    peer.write(out / "verification.json", dict(
        status="passed", seeds=list(peer.SEEDS), windows=[pca.WINDOW, WINDOW],
        min_overlap=pca.MIN_OVERLAP, pca_half_life=pca.HALF_LIFE, alpha=alpha,
        fallback_stock_days=fallback, stock_days=len(test), archived_metrics_matched=True, runs=audits))
    print(summary[[column + "_mean" for column in cols]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    peer.seed_all(100)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.out)


if __name__ == "__main__":
    main()
