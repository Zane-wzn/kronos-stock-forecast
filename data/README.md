# 归档输入契约

复现实验时在仓库根目录准备 `.peer_pca_inputs/`。已有完整实验工作区的用户可以复制该目录并保留所指向的归档，或将该目录链接到现有的完整输入。检查任何归档符号链接均能在当前机器上解析。原始市场数据、模型权重、隐藏缓存和逐股票预测存放在本地归档。

| 路径（相对 `.peer_pca_inputs/`） | 内容 |
| --- | --- |
| `{market}/hidden_cache/meta_{train,val,test}.csv` | 面板日期和股票，包含 `as_of_date`、`stock`，测试还需 `input_end_date`、`label_end_date`，CSI300 测试需由竖线分隔的 `label_dates` |
| `{market}/hidden_cache/current_{train,val,test}.npy` | 各切分原序的信号日收盘值，供滚动相关使用 |
| `{market}/hidden_cache/hidden_{train,test}.npy` | 行序与元数据严格对齐的 512 维 Kronos 表征 |
| `{market}/hidden_cache/labels_test.npy` | 与测试元数据对齐的十步收益目标，形状为 `(N, 10)` |
| `{market}/predictions/Prediction_online_seed{seed}.csv.gz` | 基线预测和真实值，列为 `pred_return_1..10`、`actual_return_1..10`，按 `as_of_date, stock` 对齐 |
| `csi300/native/seed{seed}/` | `offline/{model.pt,complete.json}`、`A/A/access_log.csv`、`A/checkpoints/A/{date}.pt`、`A/prediction_shards/A/{date}.csv` |
| `{nasdaq100,sp500}/native/seed{seed}/` | `offline/initial.pt`、`Adapter_CL/days/{date}.{json,npy}` 和 `Adapter_CL/final.pt`。Nasdaq-100 还需 `offline/complete.json`、`Adapter_CL/supervision_access.csv.gz`、`protocol.json`，S&P 500 需 `run_config.json` |
| `controls_120d/{csi300,nasdaq100}/` | `selected_alpha.json` 和各 `seed{seed}/predictions.npz`，包含 `A` 和 `all_peers_online_weighted` |
| `controls_120d/sp500/` | `verification.json`、`sp500_selected_alpha.json` 和 `sp500_seed{seed}_endpoint.npz` |
| `controls_120d/` | `metrics_by_seed.csv` 与 `alpha1/metrics_by_seed.csv`，分别供 CSI300 和美股核对 |

`market` 取 `csi300`、`nasdaq100` 或 `sp500`，`seed` 取 100、101、102。日期使用 ISO 格式，股票 ID 保留为字符串，原序行号用于预测日志和矩阵对齐。[复现指南](../docs/reproduction.md)
