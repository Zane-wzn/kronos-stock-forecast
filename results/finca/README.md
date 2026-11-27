# FinCA results

Three seeds (100–102) share each market's dates and candidate panel. Portfolio results use next-open execution, 15 basis points per side, and corporate-action accounting. Annualized return uses 252 trading sessions, with annualization applied to each seed before averaging.

| Result | Files |
| --- | --- |
| Static Adapter, Head Only and Continual Update | [Policy summary](table4_online_updates.csv), [paired intervals](tables7_8_update_intervals.csv) |
| Target-only, unprojected peers, fixed PCA and online PCA | [Peer correction summary](table5_peer_correction.csv), [paired intervals](table5_paired_intervals.csv) |
| Offline adaptation and external forecasts on shared candidates | [Baseline comparisons by seed](paper_comparisons_by_seed.csv) |
| CSI300 shared-candidate sensitivity | [Daily returns](shared_candidate_daily.csv), [5-, 10- and 20-session paired intervals](shared_candidate_intervals.csv) |
| Ten-session Endpoint prediction errors | [RMSE, MAE and directional accuracy](table6_endpoint_errors.csv) |
| Annualized portfolio curves | [Curves](figure3_annualized_curves.csv), [ten-session paired intervals](figure3_intervals.csv) |
| Inputs for metric verification | [Per-seed metrics](all_by_seed.csv), [daily portfolio returns and RankIC](all_daily.csv.gz) |

In the archived labels, `Online Head Only` corresponds to Head Only and `Full Model` corresponds to Continual Update. See the [reproduction guide](../../docs/finca-reproduction.md) for verification and replay commands.
