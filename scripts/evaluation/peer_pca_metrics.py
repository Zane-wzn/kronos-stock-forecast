"""Evaluate ten-horizon forecasts and paired differences on the retained panels."""
import numpy as np
import pandas as pd

from scripts.evaluation.evaluate_rank_scores import evaluate_signal
from scripts.evaluation.paired_statistics import block_indices
from scripts.experiments.peer_pca_replay import EXPECTED, KEYS, SEEDS, write

def evaluate(cfg, meta, actual, predictions, seed, directory):
    metrics, daily = [], []
    identity = dict(market=cfg["market"], seed=seed)
    for model, pred in predictions.items():
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        for signal, scores, base in (
            ("endpoint", pred[:, -1], predictions["A"][:, -1]),
            ("path", pred.mean(axis=1), predictions["A"].mean(axis=1)),
        ):
            values, day, _, audit = evaluate_signal(meta[KEYS], scores, actual, cfg, signal, base)
            tags = dict(**identity, model=model, signal=signal)
            metrics.append(dict(**tags, rmse_10=rmse, **values))
            daily.append(day.assign(**tags))
            write(directory / f"portfolio_audit_{model}_{signal}.json", audit)
    pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(directory / "daily_metrics.csv", index=False)


def paired(daily, method, controls, seeds=SEEDS):
    results = []
    for (market, signal), group in daily.groupby(["market", "signal"]):
        for metric in ("rankic", "net_return"):
            panel = group.pivot(index="as_of_date", columns=["model", "seed"], values=metric).sort_index()
            assert len(panel) == EXPECTED[market][0] and np.isfinite(panel.to_numpy()).all()
            for block in (10, 20):
                index = block_indices(len(panel), block)
                a = panel[method][list(seeds)].to_numpy()
                for control in controls:
                    b = panel[control][list(seeds)].to_numpy()
                    if metric == "rankic":
                        difference = a - b
                        observed = difference.mean(axis=0)
                        simulated = difference[index].mean(axis=1)
                    else:
                        observed = np.prod(1 + a, axis=0) - np.prod(1 + b, axis=0)
                        simulated = np.prod(1 + a[index], axis=1) - np.prod(1 + b[index], axis=1)
                    scopes = [("mean", observed.mean(), simulated.mean(axis=1), int((observed > 0).sum()))]
                    if block == 10:
                        scopes.extend((str(seed), observed[i], simulated[:, i], int(observed[i] > 0))
                                      for i, seed in enumerate(seeds))
                    for scope, delta, draws, wins in scopes:
                        low, high = np.quantile(draws, [.025, .975])
                        results.append(dict(market=market, signal=signal, metric=metric, reference=control,
                                            scope=scope, block_days=block, delta=delta,
                                            ci_low=low, ci_high=high, wins=wins))
    return pd.DataFrame(results)
