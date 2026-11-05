# Temporal Attention 扩大版 CPU 实验

结论：本配置的小范围实验没有支持 temporal attention 的稳定增量价值。在线 RMSE 略降，但 MAE 与横截面排序指标未改善；终点组合收益上升也几乎被等权汇总复现。

## 数据、预算与公平性

- 执行约 11.48 分钟，低于一小时预算；特征提取约 5.01 分钟。完成预设 8 轮，不为凑时间追加测试期调参。
- 固定 hash 从训练期可见股票选择 60 股，不按标签或收益选股。训练 80 个日期跨 2020-12-14—2024-12-13，验证 40 个日期跨 2025-01-02—2025-06-13；测试为 2025-07-01—2025-10-28 的 80 个连续交易日。
- 训练 4583、验证 2368、测试 4787 样本；90 日输入、10 日累计收益路径。样本存在原固定成分股/未来完整标签可评价筛选偏差，已查看测试窗口，属于回顾性开发实验。
- 冻结原 Kronos 与 tokenizer；同一个已训练 Prediction Adapter 起点；attention 2 heads/width32/零输出初始化。等权控制共享 LN、V、输出投影，冻结 Q/K；其可训练参数少于 attention，不能称容量完全匹配。
- 三种子 100/200/300；相同批次顺序、学习率、loss 和 576 次离线优化步；每轮用验证 MSE 选模，含 epoch0。所有方法均选 epoch1（72 个 batch steps），此次 Q/K 已经历非零更新，不再是前次 smoke 的单步情况。
- 静态流冻结验证选择的 checkpoint；在线流从同一 checkpoint 出发，先预测落盘再读取完整成熟反馈。每流 70 次每日更新，各 22,400 标签标量，每 horizon 最多32股；历史源标签先成熟才使用，次日启用新版本。
- 三种结构分别评价静态和在线流。MAE/RMSE 与方向使用实际路径，IC/RankIC 按日截面计算。主表为种子指标均值，不是预测集成。

## 在线结果（三种子均值）

|模型|轨迹MAE↓|轨迹RMSE↓|Path方向准确率|Path IC|Path RankIC|Endpoint IC|Endpoint RankIC|
|---|---:|---:|---:|---:|---:|---:|---:|
|adapter|0.038296|0.060339|49.32%|0.036213|0.014818|0.035149|0.015755|
|temporal_mean|0.038333|0.060312|49.14%|0.035150|0.013500|0.033517|0.013982|
|temporal_attention|0.038336|0.060314|49.15%|0.034985|0.013492|0.033278|0.013851|

Attention 相对 Adapter：MAE +0.104%，RMSE -0.042%（均为误差相对变化）。方向准确率低于50%，不能说方向预测有效。

## 子股票池组合结果

仅为约60股子池 Top6/Drop1、最低持有5日、单边15bps、下一日 close-to-close 简化代理，首日现金建仓。不是完整CSI300 Top30，也不是官方开盘执行AER。各日约59–60股；成分变化强制退出规则沿用现有 evaluator。

|模型|Path净收益|Path Sharpe|Path最大回撤|Path换手|Endpoint净收益|Endpoint Sharpe|Endpoint最大回撤|
|---|---:|---:|---:|---:|---:|---:|---:|
|adapter|21.96%|2.619|-6.71%|15.90%|16.99%|2.687|-4.84%|
|temporal_mean|20.95%|2.568|-6.71%|15.90%|22.16%|3.135|-4.41%|
|temporal_attention|21.18%|2.602|-6.71%|15.83%|22.25%|3.129|-4.41%|

Endpoint收益与排序指标出现相反方向，说明Top6边界持仓变化不能概括为整体预测改善。不能只挑Endpoint净收益报成功；等权汇总收益相近，也不能将变化归因于学习attention权重。

## 静态流结果（三种子均值）

|模型|轨迹MAE|轨迹RMSE|Path IC|Path RankIC|Path净收益|Endpoint净收益|
|---|---:|---:|---:|---:|---:|---:|
|adapter|0.038263|0.060823|0.066841|0.055859|26.15%|23.70%|
|temporal_mean|0.038281|0.060920|0.064692|0.053274|25.74%|21.09%|
|temporal_attention|0.038280|0.060921|0.064654|0.053535|28.01%|21.47%|

在线效果并非对所有指标都好：本面板离线后静态的IC/RankIC高于各自在线版本，在线RMSE改善不能被描述成全面增益。

## 配对不确定性

日期为统计单位，先对种子差值平均，再做3000次非循环日期块bootstrap；block10与20。区间不重训练、不重建持仓，未校正多重检验。方向列是轨迹整体方向率差；不是Path方向率。

|在线对照|信号|指标|差值|block10 95%区间|block20 95%区间|
|---|---|---|---:|---|---|
|temporal_attention_minus_adapter|path|mae|+0.0000401|[-0.0000306, +0.0000852]|[-0.0000291, +0.0000680]|
|temporal_attention_minus_adapter|path|mse|-0.0000030|[-0.0000157, +0.0000056]|[-0.0000150, +0.0000020]|
|temporal_attention_minus_adapter|path|rankic|-0.0013262|[-0.0024831, -0.0003304]|[-0.0020443, -0.0004732]|
|temporal_attention_minus_adapter|endpoint|rankic|-0.0019036|[-0.0028371, -0.0007774]|[-0.0027436, -0.0006017]|
|temporal_attention_minus_temporal_mean|path|mae|+0.0000029|[+0.0000006, +0.0000053]|[+0.0000007, +0.0000048]|
|temporal_attention_minus_temporal_mean|endpoint|rankic|-0.0001303|[-0.0003745, +0.0002688]|[-0.0003831, +0.0002982]|
|temporal_attention_minus_temporal_mean|endpoint|top_net|+0.0000079|[-0.0001876, +0.0001370]|[-0.0001834, +0.0001552]|

## 实现核验与后续判断

- 8项模块/实际引擎测试通过；未来未成熟标签置NaN不改变在线参数和预测；预测落盘hash、同反馈事件、一次消费、次日版本、同测试键审计通过（audit.json）。
- 重新生成11,738条完整float32序列，不使用旧的截断训练hidden文件；cache中Kronos/tokenizer训练前后状态hash一致。
- learned_attention_diagnostics.json保留Q/K权重变化、熵和平均权重；attention_weights.json包含在线首尾权重。
- 目前不建议直接扩大attention层数或据此宣称方法有效。若继续，先研究为何MAE/排序下降而RMSE略改善，并用同容量MLP、同预算调优普通头及未查看时段确认；本次没有这些确认性证据。

## 文件索引

- comparison_metrics.csv / seed_summary.csv：所有种子、静态/在线、Path/Endpoint指标。
- horizon_metrics_extended.csv：各预测期限MAE/RMSE、方向率、平衡方向率。
- evaluation/*/evaluation_summary.json：F1、MCC、零收益基线R²、Newey-West、分位差和全部原evaluator输出。
- paired_date_bootstrap.csv：block10/20配对日期区间。
- original_checkpoint_evaluation.json：未做离线再适配的原始checkpoint参考。
- label_access_log.csv / online_update_log.csv / issued/：成熟标签、预测先落盘与更新轨迹。
- run_config.json / source_manifest.json / sequence_cache_manifest.json：输入、源码、权重身份。

![净收益与RankIC曲线](comparison.png)
