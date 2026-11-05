# HCA 增益来源归因实验

20条完整在线流：2 checkpoint × 2 Adapter状态 × 5种策略。每条226日、66,103个预测键、216次更新、69,120个成熟标量标签。

## 结果

|Checkpoint|Adapter|Signal|A 净收益|HCA 净收益|HCA−A (pp)|FrozenHead 净收益|MatchedStep 净收益|Projected 净收益|A IC|HCA IC|
|---|---|---|---|---|---|---|---|---|---|---|
|Prediction|Live|path|72.357%|75.617%|+3.260|79.920%|69.087%|68.113%|0.069215|0.069050|
|Prediction|Live|endpoint|80.289%|85.115%|+4.826|66.960%|78.267%|72.681%|0.063614|0.063580|
|Prediction|Frozen|path|78.242%|72.382%|-5.859|65.662%|66.472%|80.569%|0.066534|0.067440|
|Prediction|Frozen|endpoint|66.568%|77.575%|+11.007|55.974%|80.518%|65.470%|0.060433|0.062002|
|RankIC|Live|path|41.333%|40.500%|-0.833|36.375%|35.762%|37.405%|0.038524|0.036102|
|RankIC|Live|endpoint|42.890%|41.616%|-1.274|40.750%|51.676%|41.994%|0.046365|0.040383|
|RankIC|Frozen|path|38.404%|39.827%|+1.423|34.480%|35.832%|37.561%|0.035906|0.033903|
|RankIC|Frozen|endpoint|43.583%|38.079%|-5.504|51.683%|39.830%|40.124%|0.043187|0.038155|

## 数据支持的描述

- 有效头更新范数：HCA/A 的均值比为 1.395—1.408；PCA子空间内为 1.426—1.442，子空间外为 0.997—1.000。这直接支持 HCA 主要放大已有 PCA 方向更新的机制描述。
- 幅度匹配后，与 HCA 的逐日 Path 信号排序相关性为 0.999504—0.999527，每日无持仓约束 Top30 集合平均重合率为 98.04%—98.24%。均值包含10日共同暖启动；集合不是实际持仓。幅度是预测排序变化的重要解释，但仍有残余差异，不能认定全部收益差由此产生。
- HCA 相对 A 的四个配置 Path 收益差95%区间均跨零；收益差方向随 checkpoint 和 Adapter 状态改变。当前没有稳定、可推广的 HCA 净收益增益证据。
- 冻结原头后，Prediction/Live 的 Path 收益升至79.920%，但 Endpoint 降至66.960%；RankIC/Live 的 IC升高而收益降低。该干预暴露指标冲突，不能作为更优最终模型直接选用。
- Prediction/Live：HCA−A 的 Path 净收益差为 +3.260 个百分点，95%配对区间 [-8.577, +20.412]；IC 差 -0.000165。
- 同一配置下，幅度匹配对照相对 HCA 的收益差为 -6.530 个百分点，IC 差为 -0.000028。
- 同一配置下，方向投影对照相对 HCA 的收益差为 -7.504 个百分点，IC 差为 +0.000002。
- 同一配置下，冻结原头对照相对 HCA 的收益差为 +4.303 个百分点，IC 差为 -0.000537。
- Prediction/Frozen：HCA−A 的 Path 净收益差为 -5.859 个百分点，95%配对区间 [-19.497, +11.046]；IC 差 +0.000906。
- 同一配置下，幅度匹配对照相对 HCA 的收益差为 -5.910 个百分点，IC 差为 -0.000013。
- 同一配置下，方向投影对照相对 HCA 的收益差为 +8.187 个百分点，IC 差为 -0.000925。
- 同一配置下，冻结原头对照相对 HCA 的收益差为 -6.720 个百分点，IC 差为 -0.005849。
- RankIC/Live：HCA−A 的 Path 净收益差为 -0.833 个百分点，95%配对区间 [-16.215, +18.771]；IC 差 -0.002422。
- 同一配置下，幅度匹配对照相对 HCA 的收益差为 -4.738 个百分点，IC 差为 -0.000502。
- 同一配置下，方向投影对照相对 HCA 的收益差为 -3.095 个百分点，IC 差为 +0.002194。
- 同一配置下，冻结原头对照相对 HCA 的收益差为 -4.125 个百分点，IC 差为 +0.010042。
- RankIC/Frozen：HCA−A 的 Path 净收益差为 +1.423 个百分点，95%配对区间 [-13.430, +14.674]；IC 差 -0.002003。
- 同一配置下，幅度匹配对照相对 HCA 的收益差为 -3.996 个百分点，IC 差为 -0.000456。
- 同一配置下，方向投影对照相对 HCA 的收益差为 -2.266 个百分点，IC 差为 +0.001888。
- 同一配置下，冻结原头对照相对 HCA 的收益差为 -5.348 个百分点，IC 差为 +0.009385。

## 对照的准确含义

- HCA 的线性 CoeffHead 可折叠进原线性头，函数类没有扩大；这些对照检验优化路径和参数化的作用。
- A_MatchedStep 仅匹配有效头（权重及偏置合并）的每日更新范数；Adapter 和 Adam 状态沿自身轨迹演化，不等价于匹配全部模型或学习率。
- A_ProjectedUpdate 对 AdamW 候选头更新施加 I+BBᵀ 后恢复原范数；它不是 HCA 在 AdamW 下的精确等价算法。
- FrozenHead 与 A 的差异同时涉及冻结原头和新增残差分支；该组不能单独证明 CoeffHead 的独立净贡献。
- Live/Frozen 改变 Adapter 可训练性，也改变参与全局梯度裁剪的参数集合；因此是完整策略的干预比较。
- 对照结果接近或差值区间跨零，不证明两种方法等价，也不能据此判定唯一增益来源。

## 完整窗口 Path 配对区间

|comparison|metric|delta|95%low|95%high|
|---|---|---:|---:|---:|
|Prediction_Live_HCA_minus_Prediction_Live_A|mean_ic|-0.000165|-0.004053|+0.003350|
|Prediction_Live_HCA_minus_Prediction_Live_A|net_cumulative_return|+0.032596|-0.085771|+0.204117|
|Prediction_Live_HCA_FrozenHead_minus_Prediction_Live_A|mean_ic|-0.000702|-0.007413|+0.006609|
|Prediction_Live_HCA_FrozenHead_minus_Prediction_Live_A|net_cumulative_return|+0.075623|-0.190382|+0.382418|
|Prediction_Live_HCA_minus_Prediction_Live_HCA_FrozenHead|mean_ic|+0.000537|-0.010338|+0.010458|
|Prediction_Live_HCA_minus_Prediction_Live_HCA_FrozenHead|net_cumulative_return|-0.043027|-0.355581|+0.292190|
|Prediction_Live_HCA_minus_Prediction_Live_A_MatchedStep|mean_ic|+0.000028|-0.000390|+0.000398|
|Prediction_Live_HCA_minus_Prediction_Live_A_MatchedStep|net_cumulative_return|+0.065300|-0.024151|+0.200662|
|Prediction_Live_HCA_minus_Prediction_Live_A_ProjectedUpdate|mean_ic|-0.000002|-0.003627|+0.003378|
|Prediction_Live_HCA_minus_Prediction_Live_A_ProjectedUpdate|net_cumulative_return|+0.075036|-0.044511|+0.277460|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A|mean_ic|+0.000906|-0.002569|+0.004229|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A|net_cumulative_return|-0.058594|-0.194970|+0.110464|
|Prediction_Frozen_HCA_FrozenHead_minus_Prediction_Frozen_A|mean_ic|-0.004943|-0.012317|+0.003254|
|Prediction_Frozen_HCA_FrozenHead_minus_Prediction_Frozen_A|net_cumulative_return|-0.125794|-0.389785|+0.095398|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_HCA_FrozenHead|mean_ic|+0.005849|-0.005520|+0.016175|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_HCA_FrozenHead|net_cumulative_return|+0.067201|-0.186144|+0.389645|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A_MatchedStep|mean_ic|+0.000013|-0.000433|+0.000409|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A_MatchedStep|net_cumulative_return|+0.059103|-0.026493|+0.188291|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A_ProjectedUpdate|mean_ic|+0.000925|-0.002302|+0.004105|
|Prediction_Frozen_HCA_minus_Prediction_Frozen_A_ProjectedUpdate|net_cumulative_return|-0.081866|-0.208074|+0.086848|
|Prediction_Live_HCA_minus_Prediction_Frozen_HCA|mean_ic|+0.001610|-0.003850|+0.007273|
|Prediction_Live_HCA_minus_Prediction_Frozen_HCA|net_cumulative_return|+0.032347|-0.106856|+0.199742|
|Prediction_Live_HCA_FrozenHead_minus_Prediction_Frozen_HCA_FrozenHead|mean_ic|+0.006922|-0.003704|+0.017001|
|Prediction_Live_HCA_FrozenHead_minus_Prediction_Frozen_HCA_FrozenHead|net_cumulative_return|+0.142574|-0.067488|+0.448945|
|RankIC_Live_HCA_minus_RankIC_Live_A|mean_ic|-0.002422|-0.006067|+0.001394|
|RankIC_Live_HCA_minus_RankIC_Live_A|net_cumulative_return|-0.008331|-0.162149|+0.187707|
|RankIC_Live_HCA_FrozenHead_minus_RankIC_Live_A|mean_ic|+0.007620|-0.001481|+0.016424|
|RankIC_Live_HCA_FrozenHead_minus_RankIC_Live_A|net_cumulative_return|-0.049579|-0.219720|+0.133784|
|RankIC_Live_HCA_minus_RankIC_Live_HCA_FrozenHead|mean_ic|-0.010042|-0.021698|+0.002357|
|RankIC_Live_HCA_minus_RankIC_Live_HCA_FrozenHead|net_cumulative_return|+0.041249|-0.219650|+0.326555|
|RankIC_Live_HCA_minus_RankIC_Live_A_MatchedStep|mean_ic|+0.000502|+0.000013|+0.000934|
|RankIC_Live_HCA_minus_RankIC_Live_A_MatchedStep|net_cumulative_return|+0.047378|-0.019992|+0.156419|
|RankIC_Live_HCA_minus_RankIC_Live_A_ProjectedUpdate|mean_ic|-0.002194|-0.005662|+0.001467|
|RankIC_Live_HCA_minus_RankIC_Live_A_ProjectedUpdate|net_cumulative_return|+0.030952|-0.105992|+0.223046|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A|mean_ic|-0.002003|-0.005534|+0.001774|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A|net_cumulative_return|+0.014228|-0.134302|+0.146735|
|RankIC_Frozen_HCA_FrozenHead_minus_RankIC_Frozen_A|mean_ic|+0.007383|-0.002548|+0.017025|
|RankIC_Frozen_HCA_FrozenHead_minus_RankIC_Frozen_A|net_cumulative_return|-0.039249|-0.189477|+0.138101|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_HCA_FrozenHead|mean_ic|-0.009385|-0.021760|+0.003613|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_HCA_FrozenHead|net_cumulative_return|+0.053477|-0.193396|+0.257411|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A_MatchedStep|mean_ic|+0.000456|-0.000010|+0.000872|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A_MatchedStep|net_cumulative_return|+0.039956|-0.032209|+0.149886|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A_ProjectedUpdate|mean_ic|-0.001888|-0.005278|+0.001761|
|RankIC_Frozen_HCA_minus_RankIC_Frozen_A_ProjectedUpdate|net_cumulative_return|+0.022663|-0.131006|+0.161863|
|RankIC_Live_HCA_minus_RankIC_Frozen_HCA|mean_ic|+0.002199|-0.001074|+0.005430|
|RankIC_Live_HCA_minus_RankIC_Frozen_HCA|net_cumulative_return|+0.006727|-0.105176|+0.155269|
|RankIC_Live_HCA_FrozenHead_minus_RankIC_Frozen_HCA_FrozenHead|mean_ic|+0.002856|-0.001550|+0.007127|
|RankIC_Live_HCA_FrozenHead_minus_RankIC_Frozen_HCA_FrozenHead|net_cumulative_return|+0.018955|-0.103869|+0.137918|

## 限制

meta.csv 原文件缺失，已按原 build_cache 的列读取与 row_id 分配规则从 SHA256 核验通过的原始 eligible_stock_date_pairs.csv 恢复；恢复依据、哈希及标签对齐误差见 metadata_recovery.json。此前置恢复后，四条历史在线预测复现误差均为0。 这是已经查看过的 CSI300 固定面板上的回顾性机制实验，不能充当独立确认；单种子、单市场，未做多重比较校正。沿用历史成分面板，保留潜在成分前视/幸存者偏差。所有组使用完整成熟路径标签，不检验 prefix feedback。A_MatchedStep 使用同日 HCA 更新范数，是 oracle 诊断对照。Bootstrap 重采样已实现日收益，不重训或重放持仓。post_warmup_216 为同一226日组合的后216日切片，不重新建仓。

## 配置

{
  "variants": [
    "A",
    "HCA",
    "HCA_FrozenHead",
    "A_MatchedStep",
    "A_ProjectedUpdate"
  ],
  "adapter_modes": [
    "Live",
    "Frozen"
  ],
  "seed": 100,
  "window": "2025-07-01/2026-06-05",
  "updates_per_stream": 216,
  "labels_per_stream": 69120,
  "rank": 2,
  "basis": "train_split_PCA",
  "head_lr": 0.0001,
  "adapter_lr_live": 1e-05,
  "optimizer": "AdamW wd0",
  "clip_norm": 1,
  "loss": "SmoothL1 beta1 mean horizons",
  "dropout": false,
  "cost_rate": 0.0015,
  "top_k": 30,
  "drop_n": 3,
  "min_holding_days": 5,
  "bootstrap": "block10 reps5000 seed20261013"
}
