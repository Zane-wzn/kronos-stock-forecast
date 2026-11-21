# Paper Reading

围绕三个问题阅读文献。冻结的个股基础表征提供什么信息？其他股票能为横截面排序补充什么信息？标签延迟时如何更新预测模型？

| 方向 | 文献 | 阅读重点与本项目关系 |
| --- | --- | --- |
| 金融基础模型 | *Kronos: A Foundation Model for the Language of Financial Markets* | 关注 K 线表征及其跨资产预训练。当前实验使用冻结的 512 维个股市况特征，并记录各日期的可观测边界。[论文](https://arxiv.org/abs/2508.02739) |
| 市场条件化交互 | *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*, AAAI 2024 | 比较按时点进行股票间注意力与对同日 Adapter 表征做后置聚合。原版市场引导模块和预测期与本项目的 90→10 设定不同。[出版页](https://ojs.aaai.org/index.php/AAAI/article/view/27767) |
| 价量动态关系 | *Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction*, CIKM 2022 | 对照滚动相关构图、正负关系及相关窗口。当前实现用正 Pearson 权重并在权重退化时等权。[出版页](https://doi.org/10.1145/3511808.3557089) |
| 学习股票间交互 | *Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts*, KDD 2021 | 关注端到端学习的动态关系与市场上下文。适合与显式相关加权比较。[出版页](https://doi.org/10.1145/3447548.3467297) |
| 轻量关系建模 | *StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting*, AAAI 2024 | 对比可学习的股票与市场信息混合和基于滚动相关的显式同伴聚合。关注训练成本与股票池变化。[出版页](https://ojs.aaai.org/index.php/AAAI/article/view/28681) |
| 关系与排序 | *Temporal Relational Ranking for Stock Prediction*, TOIS 2019 | 关注预测损失与选股排序的关系，也关注行业及供应链等外部关系与价量关系的输入差异。[出版页](https://doi.org/10.1145/3309547) |
| 在线适应 | *DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting*, KDD 2023 | 关注分布变化下的数据及模型适配。比较更新预算、标签可用性与固定面板的滚动评估。[论文](https://arxiv.org/abs/2306.09862) |

## 阅读后需要区分的实验问题

- **市场共同波动与个股差异。** 相关加权可能主要吸收共同因子。比较等权留一、正相关加权和去市场收益后的相关权重，才能定位额外排序信息的来源。
- **关系适应与参数适应。** 当前主线把 PCA 状态更新与成熟标签驱动的 Adapter 更新组合在同一流程。将静态和在线 Adapter 分别与静态、在线 PCA 交叉比较，可以估计两种更新的贡献。
- **排序与组合收益。** 固定阈值附近的排名变化会影响换仓，平均 RankIC 与净累计收益可以沿不同方向变化。已有结果与待验证论断见 [实验报告](../results/README.md)和[写作草稿](../paper_writing/README.md)。
