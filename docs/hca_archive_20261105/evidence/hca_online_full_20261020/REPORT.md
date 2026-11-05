# HCA 完整226日回顾性实验

2025-07-01—2026-06-05，66,103个股票日期键；全部模型同窗口、同成本、同持仓规则。
Kronos冻结，Adapter/head按A联合训练，HCA新增零初始化Linear(512,2)及冻结basis；只用完整成熟标签。

|模型|信号|累计净收益|IC|RankIC|Sharpe|最大回撤|换手率|
|---|---|---:|---:|---:|---:|---:|---:|
|Prediction_Static|path|62.477%|0.051746|0.046458|2.859|-8.475%|10.221%|
|Prediction_Static|endpoint|54.091%|0.036489|0.036284|2.751|-9.008%|10.221%|
|Prediction_A|path|72.357%|0.069215|0.057865|2.609|-11.767%|10.251%|
|Prediction_A|endpoint|80.289%|0.063614|0.049041|2.944|-10.638%|10.251%|
|Prediction_HCA_PCA|path|75.617%|0.069050|0.058550|2.642|-12.519%|10.251%|
|Prediction_HCA_PCA|endpoint|85.115%|0.063580|0.048807|2.992|-10.573%|10.236%|
|Prediction_HCA_Random|path|71.577%|0.069047|0.057871|2.616|-12.021%|10.251%|
|Prediction_HCA_Random|endpoint|86.575%|0.064936|0.049498|3.014|-11.042%|10.236%|
|RankIC_Static|path|39.059%|0.046783|0.035744|1.918|-9.806%|10.236%|
|RankIC_Static|endpoint|43.461%|0.047338|0.038545|2.162|-8.338%|10.236%|
|RankIC_A|path|41.333%|0.038524|0.024988|1.898|-11.519%|10.236%|
|RankIC_A|endpoint|42.890%|0.046365|0.028924|1.920|-11.364%|10.236%|
|RankIC_HCA_PCA|path|40.500%|0.036102|0.023610|1.812|-13.567%|10.236%|
|RankIC_HCA_PCA|endpoint|41.616%|0.040383|0.023580|1.807|-12.036%|10.236%|
|RankIC_HCA_Random|path|40.339%|0.038424|0.024922|1.845|-11.961%|10.236%|
|RankIC_HCA_Random|endpoint|42.476%|0.044427|0.026565|1.885|-11.036%|10.236%|

|对比A|Path收益差(百分点)|Path IC差|判定|
|---|---:|---:|---|
|Prediction_HCA_PCA|+3.260|-0.000165|partial_metric_conflict|

Prediction_HCA_PCA checks: `{"path_return_improves": true, "path_ic_not_lower": false, "endpoint_return_and_ic_not_lower": false, "drawdown_within_5pp": true, "both108_halves_return_or_ic_improves": true, "pca_return_and_ic_at_least_random_one_strict": true}`
|Prediction_HCA_Random|-0.780|-0.000168|no_consistent_positive_optimization|

Prediction_HCA_Random checks: `{"path_return_improves": false, "path_ic_not_lower": false, "endpoint_return_and_ic_not_lower": true, "drawdown_within_5pp": true, "both108_halves_return_or_ic_improves": false}`
|RankIC_HCA_PCA|-0.833|-0.002422|no_consistent_positive_optimization|

RankIC_HCA_PCA checks: `{"path_return_improves": false, "path_ic_not_lower": false, "endpoint_return_and_ic_not_lower": false, "drawdown_within_5pp": true, "both108_halves_return_or_ic_improves": false, "pca_return_and_ic_at_least_random_one_strict": false}`
|RankIC_HCA_Random|-0.994|-0.000099|no_consistent_positive_optimization|

RankIC_HCA_Random checks: `{"path_return_improves": false, "path_ic_not_lower": false, "endpoint_return_and_ic_not_lower": false, "drawdown_within_5pp": true, "both108_halves_return_or_ic_improves": false}`

## 配对区间

|对比|信号|指标|差值|95%下限|95%上限|
|---|---|---|---:|---:|---:|
|Prediction_HCA_PCA_minus_Prediction_A|path|mean_ic|-0.000165|-0.004053|+0.003350|
|Prediction_HCA_PCA_minus_Prediction_A|path|net_cumulative_return|+0.032596|-0.085771|+0.204117|
|Prediction_HCA_PCA_minus_Prediction_A|endpoint|mean_ic|-0.000034|-0.005107|+0.004633|
|Prediction_HCA_PCA_minus_Prediction_A|endpoint|net_cumulative_return|+0.048263|-0.140067|+0.285351|
|Prediction_HCA_Random_minus_Prediction_A|path|mean_ic|-0.000168|-0.000455|+0.000093|
|Prediction_HCA_Random_minus_Prediction_A|path|net_cumulative_return|-0.007803|-0.107653|+0.073049|
|Prediction_HCA_Random_minus_Prediction_A|endpoint|mean_ic|+0.001322|-0.001966|+0.004450|
|Prediction_HCA_Random_minus_Prediction_A|endpoint|net_cumulative_return|+0.062862|-0.101745|+0.289376|
|Prediction_HCA_PCA_minus_Prediction_HCA_Random|path|mean_ic|+0.000002|-0.003731|+0.003355|
|Prediction_HCA_PCA_minus_Prediction_HCA_Random|path|net_cumulative_return|+0.040399|-0.077784|+0.225382|
|Prediction_HCA_PCA_minus_Prediction_HCA_Random|endpoint|mean_ic|-0.001356|-0.003468|+0.000540|
|Prediction_HCA_PCA_minus_Prediction_HCA_Random|endpoint|net_cumulative_return|-0.014599|-0.153651|+0.119588|
|RankIC_HCA_PCA_minus_RankIC_A|path|mean_ic|-0.002422|-0.006067|+0.001394|
|RankIC_HCA_PCA_minus_RankIC_A|path|net_cumulative_return|-0.008331|-0.162149|+0.187707|
|RankIC_HCA_PCA_minus_RankIC_A|endpoint|mean_ic|-0.005982|-0.010214|-0.001129|
|RankIC_HCA_PCA_minus_RankIC_A|endpoint|net_cumulative_return|-0.012742|-0.163885|+0.130862|
|RankIC_HCA_Random_minus_RankIC_A|path|mean_ic|-0.000099|-0.000309|+0.000147|
|RankIC_HCA_Random_minus_RankIC_A|path|net_cumulative_return|-0.009941|-0.066222|+0.059440|
|RankIC_HCA_Random_minus_RankIC_A|endpoint|mean_ic|-0.001938|-0.005311|+0.002077|
|RankIC_HCA_Random_minus_RankIC_A|endpoint|net_cumulative_return|-0.004143|-0.151295|+0.124025|
|RankIC_HCA_PCA_minus_RankIC_HCA_Random|path|mean_ic|-0.002322|-0.005930|+0.001419|
|RankIC_HCA_PCA_minus_RankIC_HCA_Random|path|net_cumulative_return|+0.001610|-0.165612|+0.201388|
|RankIC_HCA_PCA_minus_RankIC_HCA_Random|endpoint|mean_ic|-0.004044|-0.005784|-0.002450|
|RankIC_HCA_PCA_minus_RankIC_HCA_Random|endpoint|net_cumulative_return|-0.008599|-0.127456|+0.143055|

evaluate：Top30/Drop3、最低持有5日、单向15bps、actual_return_1日收益、年化252；IC为逐日截面Pearson均值。
half_metrics.csv为216日辅助组合持仓收益按前108/后108日切片，不重新建仓；完整226日主结果包含10日共同暖启动。
bootstrap block10、5000次，共享日期块，重采样已实现日收益，不重训练/回放持仓，未做多重比较校正。
Endpoint守卫保守定义为收益和IC均不下降；PCA结构守卫定义为Path收益/IC均不低于Random且至少一项严格更高。
这是已经查看的固定CSI300面板的回顾性验证，保留成分股前视/幸存者偏差；不能作为独立确认、稳定收益或论文最终证据。
只验证complete-label结构化残差适配，不证明短期标签提前改善长期预测。失败后不叠加MOG/SVA。

![累计净收益](cumulative_net_returns.png)

![20日均值IC](ic_curves.png)
