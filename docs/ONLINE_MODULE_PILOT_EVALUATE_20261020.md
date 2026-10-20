# 在线模块小实验：统一 evaluate 实现

本实验比较同一 main D-RankIC checkpoint 的 S/F/A/D；入口为
`scripts/experiments/run_online_module_pilot.py`，旧 S/F/P 实验不改写。

## 固定协议

- 预测：2025-07-01—2025-09-22，完整锁定 CSI300 面板。
- 主评分：前10日共同暖启动，2025-07-15起的50日；组合从现金开始，首日换手1并计入15bps成本。
- 次要结果：完整60日，从现金独立回测；不混入主表。
- 前后25日：切分主评分组合的已实现收益，不重新建仓。
- F/A/D 的训练股票、各horizon样本键完全匹配上一轮F：原训练期可用性加固定hash预选60只（旧面板覆盖59只），每日各horizon最多32只，保留原F/P起点股票交集。
- 每日先持久化完整面板预测，再访问完整路径已成熟的标量标签并更新；下一日生效。每标签仅消费一次。
- F：只训练原头，lr=1e-4。A：原Adapter与头，lr分别1e-5/1e-4。D：仅头，同F加相似度权重。Kronos冻结且不重新提取hidden。
- AdamW，weight_decay=0，SmoothL1(beta=1)，每horizon样本平均再各horizon等权，每日一次step，所有可训练参数联合梯度裁剪1；所有Dropout关闭。
- D当前状态来自当天完整预测面板的初始冻结Adapter特征；各特征L2归一化后求均值，再归一化。成熟样本与状态的余弦相似度按训练批次中心化，temperature=0.1、log权重裁剪±ln2，然后除以均值。权重均值1、范围[0.25,4]，零状态回退均匀权重。

## 评测与判断

运行原 `python -m evaluate` CLI，不修改其实现；Top30/Drop3、最低持有5日、单边15bps、年化252、bootstrap block10/reps5000/seed100。Path=10步累计收益均值，Endpoint=第10步。两类信号独立构建组合，实际日收益仅使用 `actual_return_1`。IC=逐日横截面Pearson相关均值，另报RankIC。

主表突出累计净收益和IC。相对于F，候选须Path累计收益更高、IC不低，最大回撤恶化不超过5个百分点。将“Endpoint不明显反向”保守落实为Endpoint累计收益、IC均不低于F；前/后25日各须至少有Path收益或IC一项严格改善。收益和IC方向冲突标记部分改善，不写整体增益。

比较F−S、A−F、D−F、A−D。配对bootstrap使用同一批日期块重采样IC与已实现日收益；不重训练，不重模拟抽样路径持仓。它仅描述短窗口估计不确定性，不构成多重检验后的论文显著性结论。

## 执行与产物

```powershell
.runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_online_module_pilot.py --checks-only
.runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_online_module_pilot.py
```

重复运行用 `--output-dir` 指向新目录；非空目录拒绝覆盖。默认输出
`experiments/csi300_aligned/models/online_module_pilot_20261020/`。

正确性fixture调用真实引擎，覆盖成熟边界、一次消费、预算、预测先落盘、A零Adapter学习率/D均匀权重退化F、零学习率退化S、未来标签扰动、恢复模型及优化器续跑、同日输入顺序、权重边界。真实运行另核对旧S/F预测最大误差≤1e-7，完整面板键一致、Adapter更新/冻结、每日访问日志及分片hash。

每组保存原始无标签预测分片；全部流结束后才附加评测真实标签。checkpoint只存模型、优化器、随机状态及已消费事件，历史预测和训练访问通过日分片/journal保存，避免每个checkpoint重复保存完整预测历史。

结果包括REPORT.md、comparison_metrics.csv、half_window_metrics.csv、paired_daily_differences.csv、paired_block10_bootstrap.csv、D_weight_diagnostics.csv、正确性与真实运行审计、净收益/IC曲线、原evaluate全部指标及manifest。所有结果均为已查看时期和旧固定股票池上的回顾性探索，不保证投资收益或论文新颖性；本轮不自动叠加模块、调参或扩展窗口。
