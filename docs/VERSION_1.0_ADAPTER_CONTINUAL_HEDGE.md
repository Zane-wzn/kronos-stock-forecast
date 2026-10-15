# Version 1.0 — Adapter + Continual + Hedge

This file freezes the current reproducible experiment version under the Git tag
`v1.0-adapter-continual-hedge`.

## Frozen protocol

- Dataset: CSI300 aligned panel.
- Test window: `2025-07-01` through `2026-06-05`.
- Test coverage: 226 trading dates and 66,103 `(as_of_date, stock)` rows.
- Seed: `100`.
- Initial model: D-RankIC Adapter checkpoint, selected by validation Path RankIC.
- Continual branch: month-end mature-label updates with replay and promotion.
- Signal blend: cross-sectional Rank-Gaussian static and continual predictions.
- Hedge: initial continual weight `alpha=0.3`, history window up to 60 mature
  trading days, `eta=0.5`, RankIC weight `0.7`, error weight `0.3`, and alpha
  bounds `[0.05, 0.95]`.
- Weight updates are computed after the month-end prediction and take effect on
  the next test trading day.
- Portfolio protocol: Top30/Drop3, minimum holding period 5 trading days, and
  15 bps one-way transaction cost.

The complete protocol is documented in
[`D_RANKIC_DYNAMIC_CONTINUAL_PROTOCOL_20261013.md`](D_RANKIC_DYNAMIC_CONTINUAL_PROTOCOL_20261013.md).

## Frozen run identity

- Static output: `experiments/csi300_aligned/models/adapter_drankic_dynamic_static_20261013`
- Continual output: `experiments/csi300_aligned/models/adapter_drankic_dynamic_branch_20261013`
- Hedge output: `experiments/csi300_aligned/models/adapter_drankic_dynamic_continual_20261013`
- Initial checkpoint SHA256:
  `fe1426d1ab0e627622860fd86eb81c8f05b74fbd54df29439556050a1d813961`
- Deployment policy: no hard acceptance gate for this version; candidates are
  trained and promoted according to the registered protocol.

## Recorded results

The values below are the saved single-seed retrospective prequential evaluation
under the frozen Top30/Drop3 protocol:

| Signal | Net cumulative return | Annualized net return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Static D-RankIC | 39.06% | 44.43% | 1.918 | -9.81% |
| Continual branch | 33.70% | 38.24% | 1.877 | -9.40% |
| Dynamic Hedge | 50.72% | 58.01% | 2.396 | -10.38% |

The dynamic alpha had mean `0.2928`, minimum `0.2816`, and maximum `0.3000`
across the recorded run. These results are retrospective and use one seed on a
previously inspected fixed test period; they are a version record, not a claim
of untouched out-of-sample generalization.

## Reproduction entry points

- Run: `scripts/experiments/run_drankic_dynamic_continual.py`
- Evaluate: `scripts/evaluation/evaluate_drankic_dynamic_continual.py`
- Audit: `scripts/audit/audit_drankic_dynamic_continual.py`
- Curves: `scripts/evaluation/plot_drankic_dynamic_curves.py`

Large local artifacts such as prediction CSVs, checkpoints, hidden-state
caches, and raw panels are intentionally excluded from the Git commit. Their
content identity is recorded by the run's `run_config.json` and
`script_hashes.json` files in the local experiment output.
