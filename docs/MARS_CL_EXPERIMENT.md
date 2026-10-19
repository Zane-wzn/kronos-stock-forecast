# MARS-CL：冻结 Page1-C 与有界残差持续学习

本分支保存已经实现的 MARS-CL，不包含新的 MARS-IC 或 PROCEED 实验。
Kronos、Page1-C Prediction Adapter 和原始 `Linear(512, 10)` 预测头始终冻结；
唯一允许在线更新的是 `src/kronos_continual/page1c_mars_cl.py` 中的 `ResidualAdapter`。

## 模型与数据

- 任务：CSI300，90 日输入，未来 10 步累计收益路径。
- 测试：2025-07-01 至 2026-06-05，226 日、66,103 个股票日期键。
- 基础模型：`experiments/csi300_aligned/models/prediction_adapter/model.pt`。
- 原页面 C checkpoint SHA256：`111f72975e77e165211b441a731b4da17dca4e26629b848839ce13aaffcd94eb`。
- 原页面 C 历史净累计收益记录为 62.48%，不代表所有重新训练的 C checkpoint。
- 新适配器输入：10 维基础预测与 18 维状态摘要；六个行情字段各取 90 日均值、标准差、末值。
- 状态统计量仅在训练期拟合。结构为 `Linear(28,32) -> GELU -> Linear(32,10)`。
- 输出为 `base + 0.02 * tanh(residual)`；最后一层零初始化，初始预测等于静态基线。

## 更新协议

每 20 个测试交易日保存一组逐日预测，每 40 日在预测保存后尝试更新。
训练样本为固定历史池（训练截止前成熟，按年份抽样，上限 60,000 行）及最近最多 30 个完整成熟日期。
最新 10 个完整成熟日期单独用于 shadow 验收，不进入梯度训练。
所有在线训练和验收样本满足 `as_of_date < update_date`、`label_end_date <= update_date`。

损失为 `0.3 * SmoothL1(path) + 0.5 * pairwise_rank_loss + 0.2 * MSE(pred, base)`。
AdamW：学习率 `1e-4`、weight decay `1e-4`，每次 2 epoch，梯度裁剪 1.0。
候选复制当前模型和优化器；Path RankIC 不下降超过 0.002、路径 MSE 不超过原值 1.05 倍，
且两者至少一个严格改善时，按 `0.8 * current + 0.2 * candidate` 软写回，再次验收。
拒绝时模型和优化器不变；接受时保留候选 Adam 动量，参数使用软写回值。
Endpoint RankIC 仅记录，不参与门控。

## 运行

在仓库根目录、已安装相应设备 PyTorch 的 Python 环境中执行：

```powershell
python -m pip install -r requirements_mars_cl.txt
python scripts/experiments/run_kronos_page1c_mars_cl_csi300_90x10.py --device cpu --seed 100 --output-dir experiments/csi300_aligned/models/mars_cl_preflight --dry-run
python scripts/experiments/run_kronos_page1c_mars_cl_csi300_90x10.py --device cpu --seed 100 --output-dir experiments/csi300_aligned/models/mars_cl_run/seed100
```

需要本地准备原始行情、锁定标签缓存、完整 hidden cache、股票日期面板、checkpoint 和匹配的历史预测。
路径参数可通过 `--help` 查看。正式输出目录拒绝覆盖；续跑使用相同参数及 `--resume`。
使用独立重新训练的 C checkpoint 时，同时显式提供 `--initial-checkpoint`、
`--expected-checkpoint-sha256` 和 `--historical-predictions`。
seed100 完成验收后才允许同配置的 seed101/102 进入正式运行。

```powershell
python -m pytest -q tests/test_page1c_mars_cl.py tests/test_evaluate_dashboard.py
```

统一评测使用 `evaluate/`。`src/doubleadapt/compare.py` 的 bootstrap 和收益统计函数
是当前评测依赖，因此本分支一并保存该文件及其导入的数据工具；这不表示运行 DoubleAdapt 训练。

## 比较与产物

同一 checkpoint 配对比较 Static 与 MARS-CL；不同 C 训练 seed 分别配对后汇总指标，不能与预测集成混用。
Path/Endpoint 分开评价，Top30/Drop3、最低持有 5 日、单向成本 15bps，
组合用 `actual_return_1` 兑现日收益。Bootstrap 为非循环 moving-block，block=10、5,000 次、seed=20261013。

输出预测、模型/优化器/RNG/任务位置 checkpoint、任务样本、成熟队列、更新日志、shadow 指标、
环境与数据/代码哈希、配对指标和报告。缓存、模型权重、远程同步目录、临时脚本及大型预测产物不纳入本代码提交。

测试期已经查看，结论属于回顾性在线比较。已有三 seed 结果尚未显示稳定收益增量；
本分支用于保存可复现实现，不预设持续学习优于静态模型。
