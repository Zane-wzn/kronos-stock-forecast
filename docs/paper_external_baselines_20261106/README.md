# 论文外部方法对照数据核查

2026-11-06。范围为项目实际测试过、可用于论文外部方法比较的数据。Adapter变体和HCA内部机制消融不计作独立外部方法。此次只读取保存预测并重评，没有重新训练。

## 可使用范围

|方法|已有数据|论文使用范围|
|---|---|---|
|Chronos-2|226日、66,103键；完整十步/分位数预测、配置、版本、完成状态|可作本项目零样本基础模型基线；报告已有预训练资源和历史数据重叠等限制|
|TimesFM 3.0|同上；已用HCA的Top30/Drop3规则重新评估|同上，不能引用原Top50代理口径与HCA直接混比|
|Kronos零样本|原67,390条，按固定主面板对齐66,103键|主骨干基础模型参考；不是新增独立方法|
|DoubleAdapt-GRU|三个种子100/101/102各226日、66,103键；十步预测和标签齐全、无重复、标签误差<1e-7|新增的重要外部方法候选；可报告保存预测的重评值，但缺原运行配置/完成状态，暂不能认证完整训练协议或称严格官方复现|
|ODEStream|原67,006条；回退到信号日后公共225日、65,425键|仅历史诊断；共享模型同目标日逐股预测后立即读标签更新，不能作前一收盘统一发布截面预测的因果公平主基线|
|SSPT|仅smoke，无正式完整测试预测|不能用smoke loss充当论文对比结果|

当前本地尚未发现其他独立外部方法已经完成且可直接纳入论文主表的测试数据。MOG/SVA、CFUS、谱基、LoRA、Temporal Attention、Token Adapter等属于本项目机制或结构变体，应放消融/探索实验。不能据目录名断言其执行已通过正式审计。

## 新恢复的DoubleAdapt数值

三个种子都与HCA具有完全相同的股票日期键。统一累计收益标签；90→10；Top30、Drop3、最少持有5日、单边15bps、252日年化，组合使用actual_return_1兑现每日收益。Path为十步均值、Endpoint为第十步。净收益为累计净收益，不是年化。

|seed|轨迹RMSE|Path IC|Path RankIC|Path净收益|Endpoint净收益|
|---|---:|---:|---:|---:|---:|
|100|0.063966|0.077358|0.036209|56.122%|56.707%|
|101|0.063763|-0.007911|-0.015581|19.817%|48.493%|
|102|0.063665|0.020853|0.005898|26.275%|45.950%|
|三种子均值|0.063798|0.030100|0.008842|34.071%|50.384%|
|样本标准差|0.000153|0.043381|0.026020|19.367个百分点|5.622个百分点|

均值是各种子指标算术均值，不是预测集成或拼接累计收益。轨迹误差低于HCA-PCA的0.064335，但排序和组合收益有明显种子波动；不能只选seed100。HCA目前主比较只有seed100，不能用DoubleAdapt三种子对照声称HCA多种子稳健性。训练/选模预算和预训练资源不同，比较反映整套流程差异。

## 文件与复算

- `doubleadapt_metrics.csv`：各seed和Path/Endpoint指标。
- `doubleadapt_seed_summary.csv`：均值与样本标准差。
- `doubleadapt_recovery_audit.json`：预测来源/哈希、键与标签核验、缺失运行证据。
- `DoubleAdapt_seed*_evaluation.json`及逐日CSV：统一评估明细。
- 复算脚本：`scripts/evaluation/recover_doubleadapt_paper_metrics.py`。
- 原始预测：`experiments/csi300_aligned/models/doubleadapt_gru_90x10_suite_20261013/seed{100,101,102}/predictions.csv`。

## GitHub归档与换电脑恢复

`archives/DoubleAdapt_seed100.zip`、`DoubleAdapt_seed101.zip`、`DoubleAdapt_seed102.zip`各保存一个种子的原始预测CSV（包含真实标签）、`task_membership.npz`任务成员索引及全部现存阶段检查点。它们按源文件原字节保存，未以重新计算标签覆盖原预测。每个种子包包含18个文件。

`archives/DoubleAdapt_code_snapshots.zip`保存本机历史代码压缩包中的v2、final和historical_source三套源码快照，包含GRU/DoubleAdapt模型、运行器、测试、依赖说明和历史上游代码子集。三套代码分别保留，没有覆盖当前仓库的同名模块；当前源代码缺失的model/runner可从包中查回。源码默认配置不能作为实际运行配置的证明。`historical_verification/`保存原预检和测试日志，是历史证据，不是本次重新执行训练或测试。

检查点包含什么状态以其实际字段为准；它们被保存不代表在新机器上已验证可以独立续训。原始行情、完整训练缓存和缺失的运行配置不在种子包内。`_import/seed102`是导入副本，未重复打包。每个文件的大小和SHA256见`archive_manifest.json`。

```powershell
git clone --branch codex/a-online-adaptation-icde https://github.com/Zane-wzn/kronos-stock-forecast.git
cd kronos-stock-forecast
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python scripts/evaluation/recover_doubleadapt_paper_metrics.py --prefer-archives
```

重评脚本可直接读三个种子ZIP，并从此前已归档的Prediction_static.zip恢复统一标签参考，不依赖原始行情或本机experiments目录。上面只安装重评所需依赖；恢复历史训练环境请查看源码包中的`requirements_doubleadapt.txt`。

本次已实际执行`--prefer-archives`，三种子结果与原CSV重评一致。四个ZIP共98个成员，逐成员CRC及SHA256核验通过。

本报告纠正此前只检查顶层目录而将DoubleAdapt称作“无完整结果”的误判：完整预测确实存在；原运行配置及完成状态缺失是另外一项证据限制。
