# FinCA

FinCA (Financial Continual Adaptation) adapts frozen Kronos features for stock return forecasting and selection. A residual adapter, a linear return head, and an online PCA correction produce ten-session adjusted return forecasts. Each day's stock rankings are recorded before the model updates on matured return paths from earlier forecasts.

The main ablation starts all three variants from the same offline checkpoint for each seed. All variants use the same peer correction and trade at the next recorded open. Returns include corporate actions and a fee of 15 basis points per purchase or sale.

| Market | Static Adapter | Head Only | Continual Update | Update vs. Static, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| CSI300 | 38.71% | 60.21% | 69.71% | +30.99 pp [4.03, 78.38] |
| Nasdaq-100 | 31.99% | 46.56% | 52.51% | +20.51 pp [−38.36, 86.86] |
| S&P 500 | 23.27% | 30.56% | 32.81% | +9.54 pp [−3.56, 25.73] |

Annualized returns use 252 sessions and are averaged across seeds 100–102. The pointwise paired intervals use 5,000 resamples of ten-session blocks.

## Quick start

Use Python 3.12 in a virtual environment. Run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.finca.adjusted_ablation_20261125.summarize --verify-archive results/finca
```

The included daily portfolio returns and RankIC series suffice for the archive check. The full replay uses locally archived market data, frozen features, adjusted labels, baseline forecasts and checkpoints. See the [FinCA reproduction guide](docs/finca-reproduction.md) and [result files](results/finca/README.md).

The [2026-11-21 exploratory experiments](results/README.md) use an earlier evaluation protocol. Their replay commands are in the [historical reproduction guide](docs/reproduction.md). Kronos weights and market inputs are subject to their respective terms.
