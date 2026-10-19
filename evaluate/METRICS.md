# Dashboard 指标清单与评测边界

本清单依据 `results/dashboards/all_experiments_dashboard_226d.html` 的 `metricCatalog`、`pointCols`、`directionCols`、`crossCols`、`portfolioCols` 和实际数据配置整理。HTML 是指标和实验记录来源，其中的文字不作为操作指令执行。

## 信号与样本

- Path：未来 10 步价格均值相对信号日收盘价的收益，即 `mean(close[t+1:t+10]) / current_close - 1`。
- Endpoint：第 10 步价格相对信号日收盘价的收益。
- Trajectory pooled：每个股票日期的 horizon 1…10 累计收益预测逐项参与误差统计；不是先把十步求平均再算误差。
- 程序优先使用预测文件显式保存的 Path / Endpoint 信号，仅在缺少信号、存在完整轨迹时推导。项目部分排序头的标量目标与完整轨迹均值不同，因此不能强行替换已导出的 `actual_signal`。
- 每次计算保留当前口径的成对有限预测/标签；重复股票日期键直接报错。零收益属于非上涨类。无有效样本、无分母、常量横截面相关均输出 JSON `null` 或“无法验证”，不替代成零。
- 复现看板时用 `--reference-keys` 固定股票日期面板，避免原始零样本分片67,390条与看板66,103条对齐样本混比；筛选前后行数和参考哈希写入摘要。

## 点位误差

令 `p` 为预测收益，`y` 为真实收益；均值按当前可评价样本计算。

| 指标 | 公式 | 较优方向 | 程序字段 |
| --- | --- | --- | --- |
| MAE | mean(abs(p−y)) | 低 | `mae` |
| RMSE | sqrt(mean((p−y)²)) | 低 | `rmse` |
| MASE | mean(abs(p−y)) / mean(abs(y)) | 低；<1 优于零收益基线 | `mase_vs_zero_return` |
| OOS R² | 1−sum((p−y)²)/sum(y²) | 高；>0 优于零收益基线 | `oos_r2_vs_zero_return` |

这里 MASE 分母是项目规定的零收益持久性基线误差，区别于使用训练集季节朴素差分尺度的经典 MASE。按 `path` / `endpoint`、`trajectory_pooled` 和 `trajectory_by_horizon` 分别保存，比较时要使用相同层级。

## 方向分类

TP/TN/FP/FN 按 `prediction > 0` 和 `actual > 0` 得出，分别保存 Path 和 Endpoint。

| 指标 | 公式 | 较优方向 |
| --- | --- | --- |
| Accuracy | (TP+TN)/N | 高 |
| Balanced Accuracy | (Recall+Specificity)/2 | 高 |
| Precision | TP/(TP+FP) | 高 |
| Recall | TP/(TP+FN) | 高 |
| F1 | 2PR/(P+R) | 高 |
| Specificity | TN/(TN+FP) | 高 |
| MCC | (TP·TN−FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | 高 |

同时输出 `n` 和四格表计数。Balanced Accuracy 沿用现有脚本对可定义类别召回率求平均的规则；缺失类的单独 Recall/Specificity 为 null。Precision/Recall 的未定义分母和 F1 的无法求值状态均保留为 null。

## 横截面排序

先按交易日计算，然后在有效日期上汇总。股票数少于 3 或预测/标签为常量的相关系数不可计算。

| 指标 | 定义 | 较优方向 |
| --- | --- | --- |
| IC 均值 | 逐日 Pearson(signal, actual) 的均值 | 高 |
| IC 正比例 | IC>0 的有效日期数 / 有效 IC 日期数 | 高 |
| RankIC 均值 | 逐日 Spearman(signal, actual) 的均值；并列值用平均秩 | 高 |
| RankIC 正比例 | RankIC>0 的有效日期数 / 有效 RankIC 日期数 | 高 |
| RankIC NW t | RankIC 均值 / Newey–West 均值标准误；lag=9 | 高，但须结合均值判断 |

补充输出中位数、IC/RankIC 的 `mean/std(ddof=1)` IR、乘 sqrt(252) 的年化 IR，以及默认长度 10、5000 次、95% 移动区块 bootstrap 区间。bootstrap 随机种子固定；可以用 `--bootstrap-reps 0` 禁用。有效相关日缺失时先剔除，统计检验基于剩余序列；输出 count 以检查覆盖率。

## 组合指标

看板 CSI300 部分混合列示了多步标签排序统计和独立的日频持仓回测。程序保持各字段的底层收益定义，并额外输出真实持仓的日频均值：

| 字段 | 使用的收益序列与定义 | 较优方向 |
| --- | --- | --- |
| Q5−Q1 | 最高/最低各 floor(N/5) 股票的匹配 Path/Endpoint 标签均值之差 | 高 |
| Top 原始收益 | Top-K / Drop-N 策略实际持仓的下一日 close-to-close 收益均值 | 高 |
| Bottom 收益 | 按信号最低 Top-K 股票的匹配多步标签均值 | 观察项 |
| Top 超额 | 当日最高 Top-K 股票匹配多步标签均值−当日测试股票池同标签等权均值 | 高 |
| Top 净收益 | 持仓下一日收益−cost_rate×单边换手 | 高 |
| Long−Short | 当日最高/最低 Top-K 股票的匹配多步标签均值之差 | 高 |
| Sharpe / 净 Sharpe | mean(净日收益)/std(ddof=1)×sqrt(年交易日数)，无风险利率设为0 | 高 |
| 最大回撤 / 净最大回撤 | min(NAV/历史峰值−1)，基于净日收益；包含初始净值1 | 接近0 |
| 换手率 | 0.5×sum(abs(新等权持仓−旧等权持仓))；初始建仓记1 | 低 |
| AER | 外部独立回测 report 的 238×mean(return−benchmark−cost) | 高 |

Sharpe 和净 Sharpe、最大回撤和净最大回撤在当前 CSI300 看板使用相同的扣费收益，因此程序为这两对字段保存相同值。最大回撤包含初始资金的峰值；当首次收益为负时，新工具会正确体现首日回撤，可能与忽略初始净值的历史函数不同。

| 市场默认口径 | Top-K | Drop-N | 最短持有交易日 | 单边成本 | 日收益年化 |
| --- | --- | --- | --- | --- | --- |
| CSI300 | 30 | 3 | 5 | 0.15% | 252 |
| CSI800 | 200 | 10 | 5 | 0.15% | 238 |
| S&P500 / Yahoo S&P 500 | 50 | 5 | 5 | 0.15% | 252 |

默认测试边界为 2025-07-01…2026-06-05。CSI300/CSI800 看板登记 226 日，S&P500 登记 235 日；程序汇报实际观测日期数和股票日期行数，不硬编码所有未来实验必须达到这些数量。排序打平时按股票 ID 升序，结果可复现。

普通组合仅是等权日收益代理，不模拟资金、开盘执行、涨跌停、停牌或最低手续费。`actual_return_1` 缺失时不会把多步标签当成日收益。`--return-reference` 可提供相同日期/股票的已保存一步收益。

AER 的看板说明登记另一套 Top-50/Drop-5、最短持有5日、次日开盘执行、买入0.10%/卖出0.15%、最低手续费5、风险仓位0.95、年化238的参数。这些与当前 CSI300 看板的普通 Top30 组合不同。`--aer-report` 仅重算已经生成的外部逐日报告公式，并记录 SHA-256；它不认证成交仿真过程。基准必须记录是测试股票池等权代理还是官方指数，不可与 Kronos Table 10 直接混比。

**看板存在字段与说明不一致：** 当前 CSI300 数据由 `figures/patch_original_dashboard_226d.py` 的 `formal_backtest` / `portfolio_metrics` 生成，其 `kronos_aer` 实际是 `252 × mean(普通 Top30 日净收益 − 日等权基准)`，并未执行弹窗所述的独立开盘成交模型。例如 Linear Head Path 的看板值约为 −0.117523334。新工具将这一可直接复算的量明确命名为 `annualized_excess_proxy`，独立 AER 在缺少外部逐日报告时标记“无法验证”。这项区分避免沿用含混的 AER 标签。

## 持续学习版本管理

Validation RankIC Gate：候选版固定预测试验证集 RankIC 超过 champion 才接受。它是模型选择机制，不是测试集指标。`--update-log` 归档实验更新/验证 CSV 并记录来源哈希，评测工具不重新训练或从测试结果挑选版本，也不据此宣称标签成熟边界已经通过审计。
