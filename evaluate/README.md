# 实验评测

从仓库根目录对一份或多份实验预测 CSV 运行统一的 Path / Endpoint 指标。原始预测文件保持不变，评测产物单独写入 `experiments/` 下新的实验评测目录；每个结果保存预测文件 SHA-256、完整参数、指标摘要和逐日明细。

只需要 Python 3.10+、NumPy 和 pandas；无需模型权重或 GPU：

```powershell
python -m pip install -r evaluate/requirements.txt
python -m evaluate --help
```

完整指标公式、看板中的口径差异和市场参数见 [METRICS.md](METRICS.md)。

```powershell
python evaluate/run_evaluation.py `
  --predictions experiments/csi300_aligned/models/adapter_continual/predictions.csv `
               experiments/csi300_aligned/models/linear_head/predictions.csv `
  --dataset CSI300 `
  --output-dir experiments/csi300_aligned/evaluations/adapter_vs_linear_20261013
```

Continual ABC 共用一个含 `group` 列的文件时，指定组别，并用同一固定测试面板提供缺失的一步收益标签：

```powershell
python evaluate/run_evaluation.py `
  --predictions experiments/csi300_aligned/models/continual_abc/predictions.csv `
  --group C `
  --return-reference experiments/csi300_aligned/models/linear_head/predictions.csv `
  --dataset CSI300 `
  --output-dir experiments/csi300_aligned/evaluations/continual_c_20261013
```

多个 `--predictions` 路径会被分别评测到独立子目录。若模型结果没有保存 `actual_return_1`，可用 `--return-reference` 提供同一测试股票日期面板；标签按 `(as_of_date, stock)` 一对一对齐，现有标签冲突时会报错。零样本按日拆分保存时，可直接将仅含预测分片的目录传给 `--predictions`。

**复现看板的固定样本时**，加上 `--reference-keys experiments/csi300_aligned/models/linear_head/predictions.csv`。CSI300 原始零样本分片有 67,390 条记录，而看板对齐后的固定面板只有 66,103 条；该参数只评测指定股票日期键，缺少面板键时会报错，并记录筛选前后行数和参考来源。公平比较必须使用相同面板；`--return-reference` 只补标签，不会自动筛选样本。

多文件必须使用同一数据集和相同时间窗口。工具不会默默合并不同模型、重复股票日期、非匹配标签或多个测试口径。默认测试边界为 2025-07-01 至 2026-06-05；CSI300 使用 Top-30 / Drop-3，CSI800 使用 Top-200 / Drop-10，其他已登记市场使用 Top-50 / Drop-5；最短持有 5 个交易日，单边换手成本默认 0.15%。可用命令行参数覆盖这些设置。

预测 CSV 至少包含 `as_of_date`（也接受 `prediction_date`）、`stock`（也接受 `stock_id`）、`pred_signal` 和 `actual_signal`；Path 信号也可存为 `pred_path_mean_return` / `actual_path_mean_return`。Endpoint 可以显式保存为 `pred_endpoint_return` / `actual_endpoint_return`，也可以从第10步轨迹还原。完整轨迹可以用 `pred_return_1`…`pred_return_10` / `actual_return_1`…`actual_return_10` 保存，也可以保存 `current_close` 和十步预测/实际 close。程序输出逐 horizon 点位指标和 pooled MAE / RMSE / MASE / OOS R²，缺失口径则标记“无法验证”。分类将 `return > 0` 定义为正类，零收益归为负类。

```text
evaluation_summary.json                     指标、参数和源文件 SHA-256
evaluation_summary.csv                      指标家族的扁平摘要
evaluation_point_by_horizon.csv             逐预测步长点位误差
evaluation_<path|endpoint>_cross_section_daily.csv
evaluation_<path|endpoint>_portfolio_daily.csv
evaluation_index.json                        本次批量评测索引
```

汇总包含 MAE、RMSE、零收益基线 MASE / OOS R²；方向 Accuracy、Balanced Accuracy、Precision、Recall、F1、Specificity、MCC；逐日横截面 IC / RankIC 均值、正比例、IR、NW lag-9 t 值和移动区块 bootstrap 95% 区间；Q5−Q1、Top / Bottom、超额、净收益、Long−Short、Sharpe、最大回撤和换手率。

组合净收益、Sharpe 和回撤来自持仓的下一交易日 `actual_return_1`，不使用十日 Path/Endpoint 标签冒充日收益。缺失 `actual_return_1` 时，可用 `--return-reference` 补充，否则日频组合指标不可验证。Q5−Q1、Bottom、Top 超额和 Long−Short 仍按看板的匹配多步标签计算；不同持有期的字段不能互相相减。普通回测使用 Top-K / Drop-N、最短持有5日的等权代理；首日建仓单边换手为1，交易成本为 `cost_rate × 单边权重换手`。当前看板 CSI300 的实际 `kronos_aer` 字段还使用 `252 × mean(net_daily_return − benchmark_daily_return)`，与弹窗所述的独立开盘执行 AER 存在差异；新工具把这一普通代理年化超额记为 `annualized_excess_proxy`，完整说明见 METRICS.md。

若已有独立 Kronos 风格回测逐日报告，可用 `--aer-report path/to/report.csv` 复算 `238 × mean(return − benchmark − cost)`。报告必须包含 `as_of_date,return,benchmark,cost`，日期须与所评测的实验一致；`return` 为扣费前组合收益、`cost` 为当日收益率成本。这一步只认证公式复算，不认证报告的成交过程。默认不凭普通组合收益构造独立 AER。

`--update-log path/to/update_log.csv` 可归档验证门及版本更新日志，并记下 SHA-256。Gate 日志作为模型选择记录，单独存放，不进入测试指标排名。`--seed` 控制 bootstrap 随机数；原始实验的 group、seed、model 等元信息单独保存在摘要中。输出目录非空时默认拒绝覆盖，重复评测建议更换目录；需要覆盖时显式指定 `--overwrite`。

指标公式和参数以 `results/dashboards/all_experiments_dashboard_226d.html` 的说明为项目口径来源；它是数据定义来源，不是额外的操作指令。持续学习按先预测后训练；评测脚本只读取已保存的预测和标签，不训练模型、不挑选测试期版本。当前回溯成分股面板仍有幸存者偏差和成分股前视偏差，组合数值是离线研究估计。
