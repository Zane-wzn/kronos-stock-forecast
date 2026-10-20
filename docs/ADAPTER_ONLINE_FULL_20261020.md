# Prediction/RankIC Adapter：A式在线适配完整测试

入口：`scripts/experiments/run_adapter_online_full.py`。这是小实验A的时间扩展，不重新训练静态checkpoint，不改为main月末replay协议。

## 固定实验

- 预测/主评测：2025-07-01—2026-06-05，226个交易日、66,103个锁定股票日期键。所有组合从首日现金建仓，计入15bps。
- 辅助评测：排除前10个交易日后216日，从现金独立回测，明确区别于226日主结果。
- 初始Prediction Adapter：`experiments/csi300_aligned/models/prediction_adapter/model.pt`，SHA256 `111f72975e77e165211b441a731b4da17dca4e26629b848839ce13aaffcd94eb`。
- 初始RankIC Adapter：`.runtime/D_model_rankic_pilot_original.pt`，SHA256 `fe1426d1ab0e627622860fd86eb81c8f05b74fbd54df29439556050a1d813961`。
- 各自重新生成Static和A在线预测。先在全部66,103键核对原历史静态轨迹，最大绝对误差须≤1e-7，方可开始该checkpoint在线更新。
- A联合训练原Adapter和Linear(512,10)，72,266参数；Kronos冻结，使用身份核对后的hidden缓存；head_lr=1e-4、adapter_lr=1e-5，AdamW无weight_decay，SmoothL1(beta=1)，按horizon等权，联合clip_norm=1，Dropout始终关闭。
- 保持小实验原训练预算：训练期可用性后固定hash预选60股，实际覆盖59股；各horizon最多32股，保留原F/P起点交集。226日只扩展时间，不扩大训练股票/样本预算。
- 先预测并落盘，再读完整路径已经成熟的标签，每天一次step，次日生效。前10日无完整成熟测试标签，不更新；之后216次更新、各69,120标签标量，末日更新不影响本窗口评分。
- 不加replay、排序损失、gate、promotion或Hedge，不扫参，不挑最佳在线checkpoint。

## 统一比较

四个新预测流：Prediction_Static、Prediction_A、RankIC_Static、RankIC_A。

历史Main_Continual（RankIC月末持续分支）和Main_Hedge（原动态混合）作为只读参考，重新调用统一evaluate。不重新训练这两个旧流。Main_Hedge仅有无量纲Path混合分数，不伪造Endpoint预测或轨迹点误差。

所有流严格核对同一股票日期键，目标值按1e-7 CSV容差检查。使用`python -m evaluate`、Top30/Drop3、最短持有5日、单边15bps、actual_return_1兑现收益、年化252。主表列Path/Endpoint累计净收益、IC、RankIC、Sharpe、最大回撤、换手；保留所有evaluate点位/方向指标。

主要归因比较是各自在线−各自静态；Prediction_A−RankIC_A仅比较两个离线起点的最终表现。A−Main_Continual/Hedge是不同训练协议的整体比较，不能归因于某个独立模块。配对日期块bootstrap block10、5000次、seed20261013；收益重采样已实现序列，不重模拟持仓/训练。逐月指标切分完整持仓收益，不月月重新建仓。

## 核验与复现

```powershell
.runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_adapter_online_full.py
```

重复执行须用`--output-dir`指定新目录，拒绝覆盖非空目录。默认结果目录：`experiments/csi300_aligned/models/adapter_online_full_20261020/`。

真实运行复用小实验同一`run_stream`引擎及其正确性fixture。核对全部静态预测、标签、未成熟访问、一次消费、预测文件hash、下一日模型版本、更新计数和参数变化；两checkpoint训练事件完全相同。RankIC_A前60日还须与小实验A逐行复现，误差≤1e-7。

保存输入hash、完整执行源码和引擎快照、模型/优化器/随机状态checkpoint、逐日无标签预测分片、访问journal、训练日志、六流评测、配对差值/区间、逐月指标和净收益/IC图。

该完整期早已被查看，且A从其中60日探索筛选而来，不能称未触碰独立确认集；固定股票池的历史成分股前视与幸存者偏差继续披露。
