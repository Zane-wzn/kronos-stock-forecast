# Prediction Adapter + A：ICDE 创新候选与证据边界

本文件基于同一 CSI300 面板上的 226 日回顾性实验和公开原文检索，给出可继续验证的候选模块。A（Kronos 冻结，原 Adapter 与预测头联合更新）作为方法基础/强基线，不作为已经成立的论文创新。

## 已有结果的定位

Prediction Adapter 起点上，A 相对静态模型的 Path 净累计收益为 `62.477% -> 72.357%`，平均 IC 为 `0.051746 -> 0.069215`；Endpoint 净累计收益为 `54.091% -> 80.289%`，平均 IC 为 `0.036489 -> 0.063614`。但 Path 最大回撤由 `-8.475%` 变为 `-11.767%`，收益差的 block-bootstrap 区间跨零。RankIC Adapter 起点上，A 的 Path IC 和 RankIC 下降。因此 A 只能表述为 Prediction Adapter 起点上的正向信号，不能表述为跨 checkpoint 稳定收益。

## 文献边界

- TAFAS（arXiv:2501.04970，§2、§3，原文段落 134--159、218--220、288--309）已经使用部分已观测标签、周期性调度和 gated calibration。
- ADAPT-Z（arXiv:2509.03810，§3）已经研究特征空间调整、历史梯度和延迟反馈。
- Proceed（arXiv:2412.08435，§1、§3--§4）已经研究延迟造成的概念间隔，并用概念漂移生成参数调整。
- FAC（arXiv:2605.17250，摘要与 §3）已经提出只用成熟标签的规范协议和频域校准。
- Act-Now（arXiv:2412.00108，摘要与原文段落 127--129、165--169、245--246）已经使用 Fast/Slow Stream Buffer、部分/伪标签和完整标签。
- 延迟审计（arXiv:2610.12232，摘要、§3）明确指出 next-step 更新会泄漏，且在多数据集上适配收益不稳定；因此本项目的先预测、标签成熟后更新、下一交易日生效协议必须保留。
- ADOWIP（arXiv:2606.25068，摘要、§1）已经研究观察到的 decision-loss 优先门控和硬更新预算；简单的“按误差决定是否更新”不能单独作为新颖贡献。
- SOLID（arXiv:2310.14838，§5.2--§5.3）已经研究相似样本选择、冻结特征提取器和只更新预测层；简单余弦相似度加权也不够新。

## 本轮 HCA 完整验证结果

已按独立的 `run_hca_online_full.py` 入口完成 2025-07-01—2026-06-05 的完整 226 日 CSI300 回顾性实验。实验同时从 Prediction Adapter 和 RankIC Adapter 两个静态 checkpoint 启动，并在每个 checkpoint 上比较 S、A、HCA-PCA 和 HCA-Random。所有在线流使用相同的 216 个更新日、相同的 69,120 个成熟标签标量、相同的单步预算、相同的预测面板和 evaluate 配置；HCA 只增加 1,026 个零初始化的 CoeffHead 参数，basis 在训练 split 上拟合后冻结。

关键结果如下，百分比为累计净收益，IC 为 226 日逐日截面 Pearson 均值：

| checkpoint | 模型 | Path 收益 | Path IC | Endpoint 收益 | Endpoint IC |
|---|---|---:|---:|---:|---:|
| Prediction | A | 72.357% | 0.069215 | 80.289% | 0.063614 |
| Prediction | HCA-PCA | 75.617% | 0.069050 | 85.115% | 0.063580 |
| Prediction | HCA-Random | 71.577% | 0.069047 | 86.575% | 0.064936 |
| RankIC | A | 41.333% | 0.038524 | 42.890% | 0.046365 |
| RankIC | HCA-PCA | 40.500% | 0.036102 | 41.616% | 0.040383 |
| RankIC | HCA-Random | 40.339% | 0.038424 | 42.476% | 0.044427 |

相对 A，Prediction/HCA-PCA 的 Path 收益提高 3.260 个百分点，但 Path IC 下降 0.000165，Endpoint 收益提高 4.826 个百分点而 Endpoint IC 下降 0.000034；配对 block-10 bootstrap 的收益区间均跨零。RankIC/HCA-PCA 的 Path 收益下降 0.833 个百分点、Path IC 下降 0.002422，Endpoint 收益下降 1.274 个百分点、Endpoint IC 下降 0.005982。HCA-Random 没有形成相对 A 的一致改善。

因此，HCA-PCA 的结论是 `partial_metric_conflict`（仅 Prediction checkpoint 上的收益改善信号），HCA-Random 和两个 RankIC 变体均为 `no_consistent_positive_optimization`。本轮不能把 HCA 宣称为跨 checkpoint 的 ICDE 主创新，也不继续在 HCA 之上叠加 SVA 或 MOG。完整证据、逐日配对差值、bootstrap 区间和审计记录保存在 `experiments/csi300_aligned/models/hca_online_full_20261020/REPORT.md`、`comparison_metrics.csv`、`paired_block10_bootstrap.csv`、`audit.json` 和 `decisions.json`。

HCA 仍可作为论文中的结构化负向证据或 checkpoint-dependent 候选：它说明“训练期 PCA horizon basis + 零初始化残差头”在一个 checkpoint 上可以提高收益，但不能保证 IC，也不能迁移到 RankIC checkpoint。后续若保留该方向，必须在未查看的时间窗口、第二市场或多个 seed 上预注册验证，不能根据本轮结果重新选择 rank、basis 或更新规则。

## 首选候选：Matured Outcome Graph（MOG）

### 问题

当前每个 origin 有 10 个累计收益标签。相邻 origin 的 horizon 涉及重叠的底层日收益。因此，`(stock, origin, horizon)` 不重复消费，并不意味着底层经济结果相互独立。累计收益标签本身不是同一个标签的简单副本；本方案先转为日增量，才能讨论按 realized outcome 归组。

只读目标诊断显示：66,103 个股票起点对应 661,030 条 horizon 约束，转为日增量后涉及 68,884 个 `(stock, realization_date)`，平均每个结果对应 9.596 条约束，最多 10 条；同一结果跨起点的日对数收益最大差异为 3.12e-7（float32 舍入）。这些数字来自整个预测面板，不是在线实际训练消费量，不是方法增益证据。可复算入口：`literature/icde_adapter_a_20261020/analyze_outcome_overlap.py`。

### 方法

1. 累计收益为 `y[o,k]=C[o+k]/C[o]-1`。定义日对数增量 `q[o,k]=log(1+y[o,k])-log(1+y[o,k-1])`，`y[o,0]=0`；两项必须已观测。真实收益需满足 `y>-1`。
2. 将每条监督边表示为 `(stock, origin, horizon k, target_date)`，按真实可用日期释放；有额外数据管道延迟时使用 availability_date，而不是假定价格发生即已可用。
3. 按 `(stock, target_date)` 建立 outcome 节点。只对当天预算内、当前已经释放的边做分组，不使用将来的节点 degree，也不偷偷增加 replay。
4. `L_daily = mean_v mean_(e in E_t(v)) SmoothL1(qhat_e,q_v)`：每个 outcome 节点总权重相等，节点内边等权。在固定额外梯度预算下，对组内多起点上下文进行压缩/采样。必须比较均匀边采样、均匀节点采样和随机等量采样。
5. 最小版本先保持 A 的模型不变，只改目标组织。`qhat` 从当前模型重算过去起点的累计收益预测，再转成相邻 log 增量；预测的 `1+yhat` 使用验证期锁定的正值下限，报告触发次数。重新计算训练输出不能覆盖已经发布的历史预测文件。
6. 初次只试 `L = L_A + λ_daily L_daily`。若能证明冲突机制，再研究 `L_cons=mean_v Var_(e in E_t(v))(qhat_e)`，其中只有已成熟的边，且直接监督防止所有预测退化为相同常数。历史起点拥有不同信息集，强制它们一致可能有害，故一致性项为独立、可删除的假设。

**退化边界**：如果一个更新批次内每个节点的 degree 相同，节点平均与边平均完全等价。这时“degree-normalization”本身无效，不能包装为贡献。实际创新必须来自有约束的信息分配/压缩和跨起点反馈结构，并有独立消融支持。Act-Now 已使用增量 buffer 和部分标签一致性，必须把本方案的结果级归组、预算定义与其直接比较；目前没有完成排他性新颖性证明。

### 可检验贡献

贡献假设是：显式建模 realized outcome 与多起点预测的关系，能在固定计算与监督预算下减少相关反馈造成的冗余，并改善延迟多步适配。它不是“这些标签都相同”的事实主张，也不自动带来更好的估计；需要用有效样本量、梯度冲突、重叠率、目标去重成本和 horizon 误差实验验证。

### 必做消融

`A`、`A+daily(uniform edges)`、`A+daily(grouped outcomes)`，同一预算比较。需要另跑只改变 prefix/complete 标签调度的对照，不能把“更早收到标签”的收益归因于 graph。随机节点分组只能用于打乱归组关系的负对照，不改变真实目标或标签释放日期。

## 次选候选：Horizon-Coupled Adapter（HCA）

当前 `Linear(512,10)` 的各输出行独立，但 A 的 Adapter 本身共享；因此“短期监督无法影响长期预测”不成立。真正的假设是：显式约束 horizon 输出之间的更新结构，能比已有隐式共享更有效地传递反馈。

可保持原始 checkpoint 预测完全一致，新增零初始化残差 `delta_y[o,k] = B[k,:] a_theta(z_o)`，其中 `B` 为在训练期估计后冻结的 `10 x r` horizon basis，`a_theta` 输出 r 维系数；其末层零初始化使初始残差为零。先固定 r，只与相同参数量的独立 horizon 残差头比较；增加模块本身不是新颖性。由于 B 固定，在线不改原 checkpoint 初始函数，所有组同一初始预测。

核心是 2x2：独立/耦合 horizon × 完整成熟/逐步成熟。只有耦合结构对逐步成熟的额外增益成立，且最早可用标签时间、预算、学习率选择方式一致，才支持“跨期限反馈迁移”。必须与 TAFAS 和 Act-Now 的部分标签机制区分；低秩、共享头本身是常见结构。

## 作为可靠性模块：Shadow-Validated Adapter Update（SVA）

每天先复制当前 Adapter 和优化器状态得到 candidate，在已成熟且不参与当前梯度训练的 shadow queue 上计算 loss/IC/参数漂移，再决定部署 candidate。拒绝时同时还原模型、Adam 状态和 step 计数，事件已审查的状态与是否用于训练分别记录。shadow queue 只能使用过去已释放标签；阈值必须在时间上更早的 calibration 区间锁定。

进一步可约束候选相对当前模型的截面中心化预测变化 `delta_centered=delta_y-mean_stock(delta_y)`，减少影响排序的无约束更新。需要与简单 L2/proximal regularization 和 main promotion gate 比较。这个几何约束仅为排序扰动代理，不能保证 IC、TopK 稳定或回撤上界。若要声称 TopK 保持，可在单次更新的确定预测面板上证明充分条件：任意预测改变量≤epsilon 且第K名与K+1名分数差>2epsilon，则 TopK 边界不变；这不是未来收益保证。

SVA 适合作为可靠性模块，不能把普通 error gate 宣称为新算法，因为 ADOWIP 已有观察到的 decision-loss gate。报告接受率、拒绝率、shadow 重复选择偏差、候选尝试的完整计算成本和回撤。简单加 Student-t loss 也不是原创稳健优化：NatSR（arXiv:2601.12931，§4）已有自然梯度、Student-t 有界更新和 replay。

## 不建议作为主创新

- 单独的反馈年龄衰减：本协议只在完整 10 步成熟后更新，年龄差异有限；容易退化成调参。
- 简单状态相似度加权：D 的小规模结果没有相对 F 形成一致增益，且 SOLID 已有相似样本校准。
- Fast/Slow 双 Adapter：Act-Now 已有 FSB/SSB，除非证明 outcome graph 的唯一目标归并带来额外收益。
- 直接复现 TAFAS、ADAPT-Z、Proceed、FAC：它们应作为相关工作或外部对照，不能换名后作为原创模块。
- 直接把 Hedge、replay 或 promotion gate 包装为新方法：仓库的 main 已包含相关思想。

## 最小 ICDE 实验面板

第一阶段保持小规模，只验证一个机制：A、A+daily(uniform)、A+daily(grouped)，另保留现有 S/F 参考。先检查实际 schedule 中节点 degree 是否存在差异；若均相同，跳过无效归一化，转为测试 HCA 的 2x2 机制。两条路线不直接叠加。SVA 单独针对 A 的回撤问题验证。

Prediction Adapter 是主起点，RankIC 是必要的负向稳健性诊断；不能在后者失败后删除结果。每个机制需要参数量和计算量匹配的简单对照。

固定 `evaluate`：完整 CSI300 面板、Top30/Drop3、最短持有 5 日、15bps 单边成本、Path/Endpoint 累计净收益和 IC，block=10、5000 次 bootstrap。2025-07-01—2026-06-05 全期已经查看，只能用于探索性复核。新方法所有参数必须在更早的 train/validation 中确定；下一次独立确认应锁定未查看的未来时段/其他市场。小实验有效后，论文补至少另一个市场和两个通用数据流任务，避免金融 TopK 收益主导结论。

新增报告：原始标签标量数、唯一 `(stock,target_date)`、梯度约束条数、样本边重放次数、每个 horizon 监督量、参数量、更新次数、完整训练/评估耗时和内存。新鲜信息、梯度曝光、optimizer step 是不同预算，不能只匹配其中一个就声称完全公平。

与 A 比较时，先在 complete-only 同标签键下归因；测试 prefix 时用同为 prefix 的均匀监督对照，固定共同更新日以排除暖启动差异，再另报实际部署中从更早日期生效的结果。预测依旧先落盘、更新下一交易日生效，禁止重写历史原点预测。

## 通过标准

以 A 为主对照：MOG/HCA 必须同时改善 Path 累计净收益和平均 IC，Endpoint 不出现明显反向，最大回撤恶化不超过 5 个百分点，并在前半段和后半段都至少有一个主要指标改善。只改善单个窗口或只改善收益、同时 IC 下降，标记为部分改善；两项都不优于 A，停止叠加。

以上是筛选规则，不是显著性标准。ICDE 方法论需要机制诊断、独立确认、多种 delay（例如标签正常释放后再加 0/5/10 个交易日的管道延迟）、多 horizon、多个静态训练种子，以及相对于简单闭式校准器 RLS/ELF、TAFAS/Act-Now 或 Proceed 中最贴近者的因果对照。不必在当前小验证阶段把全部基线都跑完。

## 原文来源

- [TAFAS, arXiv:2501.04970](https://arxiv.org/abs/2501.04970)
- [ADAPT-Z, arXiv:2509.03810](https://arxiv.org/abs/2509.03810)
- [Proceed, arXiv:2412.08435](https://arxiv.org/abs/2412.08435)
- [FAC, arXiv:2605.17250](https://arxiv.org/abs/2605.17250)
- [Act-Now, arXiv:2412.00108](https://arxiv.org/abs/2412.00108)
- [Delay-aware audit, arXiv:2610.12232](https://arxiv.org/abs/2610.12232)
- [ADOWIP, arXiv:2606.25068](https://arxiv.org/abs/2606.25068)
- [SOLID, arXiv:2310.14838](https://arxiv.org/abs/2310.14838)
- [NatSR, arXiv:2601.12931](https://arxiv.org/abs/2601.12931)
- [CoRe, arXiv:2609.34638](https://arxiv.org/abs/2609.34638)：跨变量校正空间交互已有相近工作，不宜直接新增行业共享校正并声称原创。
- [DPAL, arXiv:2610.14376](https://arxiv.org/abs/2610.14376)：金融 TTA 的方向/价格轨迹联合损失已有相近工作，不宜只加方向损失作为主创新。

MOG 和 SVA 仍是待验证的研究假设，不是已验证的 ICDE 贡献。HCA 已实现并完成本轮完整 226 日实验，但结果属于已经查看的固定 CSI300 面板上的回顾性机制验证：HCA-PCA 只在 Prediction checkpoint 上出现收益改善与 IC 冲突，在 RankIC checkpoint 上为负向，HCA-Random 没有一致增益。因此当前可稳妥写入论文的结论是 A 的因果在线适配协议和 HCA 的失败/部分改善消融，而不是 HCA 已经成立的普遍方法。
