# Reproducing the FinCA experiments

Run commands from the repository root on Python 3.12. The original replay environment used NumPy 2.4.4, pandas 2.2.2, SciPy 1.17.1, PyTorch 2.5.1+cu124 and threadpoolctl 3.6.0. The ablation replay uses CPU and one numerical thread. Offline fitting used CUDA. Install the corresponding CUDA build of PyTorch if refitting on GPU.

## Verify the released results

The released data includes per-session portfolio returns and RankIC aggregated across the cross-section. Verify the 252-session annualization, per-seed aggregate metrics and 5,000-draw paired bootstrap intervals, then render the available result tables and ablation chart into a new local directory.

```bash
python -m scripts.finca.adjusted_ablation_20261125.summarize --verify-archive results/finca
```

In PowerShell, set `$env:FINCA_OUTPUT = "results/finca"` for the following command. In Bash, prefix it with `FINCA_OUTPUT=results/finca`.

```bash
python -m scripts.finca.adjusted_ablation_20261125.paper_tables runs/finca/rendered
```

Unset `FINCA_OUTPUT` after rendering so full replay writes to its own directory.

## Replay from local archives

The following paths can be set as environment variables. Relative paths are resolved from the directory where Python starts.

| Variable | Default | Contents |
| --- | --- | --- |
| `FINCA_RUNS` | `runs/finca` | Prepared panels, adjusted labels and original replay references |
| `FINCA_OUTPUT` | `$FINCA_RUNS/adjusted_ablation_20261125` | Recomputed scores, tables and curves |
| `FINCA_MARKET_DATA` | `delete/files/date` | Historical raw daily bars and supplementary validation caches |
| `FINCA_ARCHIVED_RESULTS` | `delete/files/result` | Archived baseline score files for external comparisons |

Also prepare `.peer_pca_inputs/<market>/hidden_cache/` at the repository root with archived `meta_{train,val,test}.csv`, `hidden_{train,test}.npy` and `current_test.npy`. Put `Prediction_online_seed100.csv.gz` in each market's `predictions/` folder. See the [historical data contract](../data/README.md) for panel conventions. The test metadata needs `label_start_date`, and the CSI300 replay needs the pipe-delimited `label_dates` for its horizon-specific update schedule.

Copy the following prepared artifacts into `FINCA_RUNS`, keeping the original directory names from the experiment archives.

| Relative path | Required files |
| --- | --- |
| `csi300_adjusted_20261124/` | `adjusted_close.csv.gz`, `corporate_actions.csv`, `labels_test.npy`, `labels_test_extra.npy`, `daily.csv`, `seed{100,101,102}/model.pt` |
| `us_adjusted_20261125/{nasdaq100,sp500}/` | `adjusted_close.csv.gz`, `opens.csv`, `closes.csv`, `dividends.csv`, `splits.csv`, `labels_test.npy`, `daily.csv`, `endpoint_*.csv.gz`, `seed{100,101,102}/model.pt`. S&P 500 also uses `labels_extra.npy`. |
| `csi300_next_open_20261123/` | `dasfaa_extra_features_csi300.npz` |
| `next_open_main_20261123/` | `dasfaa_extra_features_sp500.npz` |

The archive layout for raw bars is a group directory under `FINCA_MARKET_DATA` followed by one of `csi300_daily/csi300_daily`, `nasdaq100_2016_20260819/daily` or `yahoo_sp500/yahoo_sp500`. The replay matches historical forecasts, daily RankIC and returns against the archived references.

```bash
python -m scripts.finca.adjusted_ablation_20261125.replay csi300
python -m scripts.finca.adjusted_ablation_20261125.replay nasdaq100
python -m scripts.finca.adjusted_ablation_20261125.replay sp500
python -m scripts.finca.adjusted_ablation_20261125.summarize
```

## Adjusted targets and baseline comparisons

With the market inputs available, the archived preparation and fitting code runs as Python modules. For CSI300, run the reconciliation step after preparing the labels to match corporate-action events against the adjustment factors. The preparation uses the archived adjustment factors under `csi300_adjusted_20261124/data/` and the supplementary train/validation labels under `*/csi300/*/reference/hidden_cache`. The US preparation uses the saved corporate-action responses under `us_adjusted_20261125/data/` and the historical validation caches. The provider download scripts are included for provenance. Exact results depend on the original snapshots.

```bash
python -m scripts.finca.csi300_adjusted_20261124.prepare
python -m scripts.finca.csi300_adjusted_20261124.reconcile
python -m scripts.finca.csi300_adjusted_20261124.train
python -m scripts.finca.us_adjusted_20261125.prepare nasdaq100
python -m scripts.finca.us_adjusted_20261125.train nasdaq100
python -m scripts.finca.us_adjusted_20261125.prepare sp500
python -m scripts.finca.us_adjusted_20261125.train sp500
```

After the three replays, run `python -m scripts.finca.adjusted_ablation_20261125.offline_comparisons MARKET` once for each market. This re-evaluates archived external model scores on the common adjusted panel and fits or loads the frozen-feature linear heads for CSI300 and S&P 500. For these markets, keep the prepared `labels_{train,val}.npy` and validation features, or restore the historical `linear_seed{100,101,102}.pt` files under the matching market in `FINCA_OUTPUT`. The external forecasts are read from `FINCA_ARCHIVED_RESULTS/<market>/*/<run>/predictions`, where the run name contains `cost_aligned`, `nasdaq100_paper_data` or `sp500_paper_data`, respectively. The CSI300 comparison also emits the 5-, 10- and 20-session sensitivity intervals on shared candidates. Prediction files use `as_of_date`, `stock` and `pred_return_10`.

The two earlier adjustment evaluations (`csi300_adjusted_20261124.evaluate` and `us_adjusted_20261125.evaluate_adjusted`) can also regenerate the original replay references. They depend on the prior next-open runs, including the `open_daily.csv` and per-seed Endpoint score exports. The paper's external baselines use archived forecasts from the original external model runs.

| Market | Test signals | Replay sessions | Full panel stock dates | Observed ten-session targets |
| --- | --- | ---: | ---: | ---: |
| CSI300 | 2025-07-01 to 2026-06-05 | 226 | 66,223 | 66,127 |
| Nasdaq-100 | 2025-07-01 to 2026-06-05 | 235 | 23,030 | 23,030 |
| S&P 500 | 2025-07-01 to 2026-06-05 | 235 | 117,474 | 117,473 |

The shared CSI300 comparisons use 66,103 stock dates. The full panel also includes supplementary execution candidates. RankIC and prediction errors use observed labels.
