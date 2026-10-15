# D-RankIC 动态持续学习：初始 alpha 0.3

## 固定协议

本轮按用户已确认的方案执行，只有一项主配置。D-RankIC 起点使用既有 epoch 3 checkpoint `D_model_rankic.pt`，当前文件 SHA256 为 `fe1426d1ab0e627622860fd86eb81c8f05b74fbd54df29439556050a1d813961`。该哈希用于核对当前授权文件，不声称历史来源已经独立验证。

冻结 Static 重新预测完整测试集，并核对已有 `adapter_dpaired_static_D_rankic_20261012` 的 checkpoint、九个缓存文件、锁定键和日期窗口哈希，以及预测和标签。Continual version 000 与 Static 的完整十步预测差异须不超过 1e-6。

测试期 2025-07-01—2026-06-05，226 日、66,103 行、seed 100。Adapter、损失、月末更新和 replay 规则沿用 beta8，不进行阈值或窗口扫描。

## 月末训练与部署

1. 当日先使用当前模型预测。
2. 仅选 `as_of_date < update_date` 且 `label_end_date <= update_date` 的成熟标签；最近 30 个完整成熟日作为本轮 recent gate，不足 20 日跳过模型训练和推广。
3. replay 标签结束日必须严格早于 recent gate 最早标签开始日；gate 和固定 anchor 的键不能进入本轮候选训练。
4. 候选最多训练 2 epoch，batch 16、head LR 3e-4、adapter LR 1e-4、weight decay 1e-4，Huber + 0.2 同日 pairwise ranking loss，gradient clip 1。候选 epoch 仅按固定 anchor RankIC 选择。
5. **本轮取消三项硬接纳门槛**。成熟日足够并完成候选训练，就对候选进行 1 epoch 推广：已选 replay + recent gate，head LR 3e-5、adapter LR 1e-5；推广阶段不使用验证集选择 epoch 或再次 gate。
6. 推广模型从下一测试交易日起生效。gate 指标只描述推广前候选，不作为推广模型的独立泛化成绩。

保留 beta8 配置中的门槛数值便于核对固定训练协议，但 `hard_acceptance_gate=false`，这些数值不参与接纳决策。跨轮滚动 gate 可能包含之前推广使用的样本，只保证本轮候选训练隔离，不声称全期独立留出。

## 信号混合与 Hedge

每个日期内，分别将 Static 和当前 Continual 的 `pred_signal` 做横截面 Rank-Gaussian。组合分数为 `(1-alpha)*static_normalized + alpha*continual_normalized`。初始 alpha 0.3，是信号权重，不是参数插值。

月末使用过去部署时实际生成的两条预测流和已成熟标签，最近最多 60 个交易日计算损失：

`0.7*(1-mean_daily_RankIC) + 0.3*clip(mean_daily_standardized_MAE/2, 0, 1)`。

Hedge eta 0.5，权重初值 Static 0.7 / Continual 0.3，输出 alpha 限制在 [0.05,0.95]；新权重只用于下一测试交易日。当月缺少足够 recent gate 时，Hedge 仍可按可用成熟历史更新。Hedge 不对新训练候选回溯预测历史窗口。

模型更新与 Hedge 可由两条因果预测流离线重建：模型训练不依赖组合权重，而权重仅依赖当时已成熟的历史预测；重建不重新生成历史模型预测。

## 评价与产物

主对照为 Dynamic−D-RankIC Static；单列 Continual−Static 以观察底层分支是否改善。Path RankIC、Pearson IC、Q5−Q1、Top10% Path 净收益，以及 Top30/Drop3 净收益、Sharpe、最大回撤均使用相同日期和键。

Top30/Drop3：次日 `actual_return_1`，最低持有 5 日，交易成本 15 bps；累计收益曲线仅绘制该可交易日收益流。保留原 dashboard 的 Top10% Path 指标口径（10 bps 单向换手成本），其十步平均累计收益存在重叠，不能累计为可交易每日净值。MAE/RMSE 等只评价 Static、Continual 原始收益尺度预测，不评价无量纲 Rank-Gaussian 混合分数。

逐日配对差值使用共享日期块 moving-block bootstrap，block 10、5,000 次、统计 seed 20261013。Top30 复利收益、Sharpe、回撤区间重采样已实现日收益，不在每个 bootstrap 样本重建持仓。

独立输出目录：

- `adapter_drankic_dynamic_static_20261013`：重现并核对旧 D-RankIC Static。
- `adapter_drankic_dynamic_branch_20261013`：无硬门槛的训练分支、候选及部署 checkpoint、gate/replay 审计。
- `adapter_drankic_dynamic_continual_20261013`：混合信号、alpha、权重日志、配对评价、曲线及完整 dashboard。

环境沿用前轮远端 CPU：Python 3.12.4、PyTorch 2.11.0+cpu、CUDA 不可用。OpenMP 兼容变量与线程数写入 environment.json；没有将该实验记录为 GPU 运行。

代码入口为 `scripts/experiments/run_drankic_dynamic_continual.py`；评价为 `scripts/evaluation/evaluate_drankic_dynamic_continual.py`；产物审计为 `scripts/audit/audit_drankic_dynamic_continual.py`；本地高清曲线为 `scripts/evaluation/plot_drankic_dynamic_curves.py`。21 项协议单元测试和相同信号的完整评估校验已通过。

结果属于单 seed、已查看固定测试期的回顾性 prequential 比较；不据本轮结果修改权重、门槛或窗口，不作为未触碰留出样本的泛化证明。
