"""Compare 60- and 120-day all-stock PCA in the two US markets."""

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

MODE = pca.MODE
ALPHA1_ARCHIVE = peer.CONTROLS / "alpha1"
METRICS = (
    "net_cumulative_return", "sharpe", "max_drawdown", "direction_accuracy",
    "balanced_direction_accuracy", "rankic", "rankic_ir", "ic", "signal_rmse",
    "signal_mae", "annualized_volatility", "net_annualized_return", "turnover",
    "topk_excess",
)


def run_market(market, device, out, alpha, control_model, variant_model):
    is_sp500 = market == "sp500"
    cfg = peer.config(market)
    cache = cfg["cache"]
    train = pd.read_csv(cache / "meta_train.csv", dtype=str, usecols=peer.KEYS)
    val = pd.read_csv(cache / "meta_val.csv", dtype=str, usecols=[*peer.KEYS, "label_end_date"])
    test, hidden = peer.panel(cache, "test")
    labels = np.load(cache / "labels_test.npy", mmap_mode="r")
    h_train = np.load(cache / "hidden_train.npy", mmap_mode="r")
    assert h_train.shape == (len(train), 512)
    assert train.as_of_date.max() < val.as_of_date.min() <= val.label_end_date.max() < test.as_of_date.min()
    dates = sorted(test.as_of_date.unique())
    assert (len(dates), len(test)) == peer.EXPECTED[market][:2]
    panels = dict(train=train, val=val, test=test)
    stocks, weights = pca.neighbor_protocol(cfg, panels, 60)
    if is_sp500:
        _, control_weights = pca.neighbor_protocol(cfg, panels, pca.WINDOW)
        schedule = {date: (np.repeat(ids, 10), np.tile(np.arange(10), len(ids)))
                    for date, ids in peer.make_schedule(test).items() if len(ids)}
        previous = {row["seed"]: row for row in peer.read_json(peer.CONTROLS / "sp500" / "verification.json")["runs"]}
        control_alpha = peer.read_json((peer.CONTROLS / "sp500") / "sp500_selected_alpha.json")["alpha"]
    else:
        prior = peer.CONTROLS / market
        control_alpha = peer.read_json(prior / "selected_alpha.json")["selected"][MODE]["alpha"]
    prior_metrics = pd.read_csv(ALPHA1_ARCHIVE / "metrics_by_seed.csv").query("market == @market")
    prior_metrics = prior_metrics.set_index(["seed", "signal", "model"])
    day_rows = test.groupby("as_of_date", sort=True).indices
    stock_ids = test.stock.map(stocks).to_numpy(dtype=int)
    fallbacks = sum(int((weights[date][np.ix_(stock_ids[ids], stock_ids[ids])].sum(axis=1) == 0).sum())
                    for date, ids in day_rows.items())
    rows, days, audits = [], [], []

    for seed in peer.SEEDS:
        print(f"{market} seed{seed}: fitting PCA and replaying baseline", flush=True)
        source = cfg["home"] / "predictions" / f"Prediction_online_seed{seed}.csv.gz"
        frame = pd.read_csv(source, usecols=peer.KEYS + peer.PC + peer.AC,
                            dtype={"stock": str}, float_precision="round_trip")
        base = peer.align_predictions(test, frame)
        actual = peer.align_predictions(test, frame, peer.AC)
        np.testing.assert_allclose(actual, labels, atol=5e-7, rtol=0)
        if is_sp500:
            checkpoint = cfg["archive"] / f"seed{seed}" / "offline/initial.pt"
            assert peer.sha(checkpoint) == previous[seed]["checkpoint_sha256"]
            assert peer.sha(source) == previous[seed]["baseline_sha256"]
            original = peer.new_model().eval()
            original.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            encoder = original.adapter
        else:
            schedule, encoder, audit = peer.supervision(market, cfg, test, seed)
            checkpoint = Path(audit["checkpoint"])
            with np.load(prior / f"seed{seed}" / "predictions.npz") as saved:
                np.testing.assert_array_equal(saved["A"], base)
                control = base + (saved[MODE] - base) / control_alpha

        context = pca.fit_training_context(train, h_train, encoder)
        observe, corrections, counts, updates = pca.correction_observer(test, stocks, weights, context)
        observer = observe
        if is_sp500:
            observe_control, correction_control, counts_control, updates_control = pca.correction_observer(test, stocks, control_weights, context)

            def observer(date, ids, encoded, head):
                observe(date, ids, encoded, head)
                observe_control(date, ids, encoded, head)

        _, replay, versions = peer.baseline_embeddings(
            market, cfg, checkpoint, encoder, test, hidden, labels, base, schedule, device, observe=observer)
        expected = np.cumsum([0, *(date in schedule for date in dates)])
        np.testing.assert_array_equal(versions, expected[:-1])
        assert (expected[-1], sum(len(ids) for ids, _ in schedule.values())) == peer.EXPECTED[market][2:]
        assert len(updates) == len(dates) and replay["baseline_prediction_max_error"] <= 1e-7
        pca.verify_corrections(corrections, counts)
        if is_sp500:
            assert len(updates_control) == len(dates)
            pca.verify_corrections(correction_control, counts_control)
            with np.load((peer.CONTROLS / "sp500") / f"sp500_seed{seed}_endpoint.npz") as saved:
                np.testing.assert_array_equal(saved["A"], base[:, -1])
                np.testing.assert_allclose((base + control_alpha * correction_control)[:, -1],
                                           saved[MODE], atol=1e-12, rtol=0)
            control = base + correction_control
        variant = base + corrections
        if alpha != 1.0:
            control = base + alpha * (control - base)
            variant = base + alpha * (variant - base)
        predictions = {"A": base, control_model: control, variant_model: variant}
        np.savez_compressed(out / f"{market}_seed{seed}_predictions.npz", **predictions)

        for signal in ("endpoint", "path"):
            def scores(array):
                return array[:, -1] if signal == "endpoint" else array.mean(axis=1)

            reference, target = scores(base), scores(actual)
            for model, forecast in predictions.items():
                pred = scores(forecast)
                values, day, _, _ = metrics.evaluate_signal(test[peer.KEYS], pred, actual, cfg, signal, reference)
                key = seed, signal, model
                if key in prior_metrics.index and (model == "A" or alpha == 1.0):
                    for metric in ("net_cumulative_return", "sharpe", "max_drawdown", "rankic"):
                        np.testing.assert_allclose(values[metric], prior_metrics.loc[key, metric], atol=1e-12, rtol=0)
                direction = _direction_metrics(pred, target)
                error = pred - target
                values.update(direction_accuracy=direction["accuracy"],
                              balanced_direction_accuracy=direction["balanced_accuracy"],
                              signal_rmse=float(np.sqrt(np.mean(error ** 2))),
                              signal_mae=float(np.abs(error).mean()),
                              annualized_volatility=float(day.net_return.std(ddof=1) * np.sqrt(252)))
                rows.append(dict(market=market, seed=seed, signal=signal, model=model,
                                 alpha=0. if model == "A" else alpha, **{col: values[col] for col in METRICS}))
                days.append(day.assign(market=market, seed=seed, signal=signal, model=model))
        audits.append(dict(market=market, seed=seed, pca_updates=len(updates), stock_days=len(test),
                           fallback_stock_days=fallbacks, baseline_updates=int(expected[-1]), **replay))
        print(f"{market} seed{seed}: verified", flush=True)
    return rows, days, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, choices=(0.025, 1.0), default=0.025)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    alpha_tag = "alpha1" if args.alpha == 1.0 else "alpha0025"
    out = args.out or peer.RESULTS / f"us_all_peer_pca_60d_{alpha_tag}_20261121"
    out.mkdir(parents=True)
    suffix = "_alpha1" if args.alpha == 1.0 else ""
    control_model, variant_model = f"rolling_120d{suffix}", f"rolling_60d{suffix}"
    peer.seed_all(100)
    torch.set_num_threads(1)
    rows, days, audits = [], [], []
    with threadpool_limits(limits=1):
        for market in ("nasdaq100", "sp500"):
            result, daily, verification = run_market(
                market, "cuda:0", out, args.alpha, control_model, variant_model)
            rows.extend(result)
            days.extend(daily)
            audits.extend(verification)
    by_seed = pd.DataFrame(rows)
    assert len(by_seed) == 36 and not by_seed.duplicated(["market", "seed", "signal", "model"]).any()
    summary = by_seed.groupby(["market", "signal", "model"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(pair) for pair in summary.columns]
    paired = metrics.paired(pd.concat(days, ignore_index=True), variant_model, ("A", control_model))
    by_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    summary.to_csv(out / "summary_mean_std.csv")
    paired.to_csv(out / "paired_intervals.csv", index=False)
    peer.write(out / "verification.json", dict(status="passed", alpha=args.alpha, windows=[60, 120],
        min_overlap=pca.MIN_OVERLAP, pca_half_life=pca.HALF_LIFE, runs=audits))
    print(summary[[metric + "_mean" for metric in METRICS[:8]]].to_string(), flush=True)


if __name__ == "__main__":
    main()
