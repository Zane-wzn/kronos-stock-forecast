# Token Adapter：粗细token预测分支

日期：2026-11-05。此为实现与smoke验证，未替换原收益Adapter和checkpoint。

## 结构

历史价量→冻结官方tokenizer→冻结Kronos8层→逐位置Adapter(512→64→512残差)→冻结官方coarse head(512→1024)→采样粗token→冻结官方DependencyAwareLayer→冻结fine head(512→1024)→采样细token→官方Tokenizer decoder→价量反归一化→收盘价收益/排序。

沿用Adapter结构，训练参数67,136；不再使用原512→10收益头。没有重新定义粗细token，严格沿用官方codebook。所有原Kronos/tokenizer/head/dep_layer冻结。

两个起点：identity新Adapter上投影为零，初始生成与官方完全一致；return_checkpoint显式加载已有Prediction Adapter权重，不加载其收益头。旧权重是面向收益的变换，不能保证适合官方token heads。

## 训练与时间边界

每条样本90日历史+10日未来，共100个粗token与细token。输入tokens[:99]，target=tokens[1:100]；只在位置89..98监督未来10个token。历史跨日teacher forcing用于并行前缀训练；评估必须free-running生成，不能使用真实未来前缀。

细token默认条件于预测分布采样的粗token（detach采样），不使用真实目标粗token。损失=(coarse CE+fine CE)/2，梯度通过冻结分类头/依赖层传到Adapter。冻结网络eval不等于no_grad，依赖层必须保留输入梯度。

归一化均值方差只取历史90日；未来用于标签不参与统计。tokenizer因果性通过未来扰动核验。禁止对所有100日做归一化；禁止用真实未来token替换生成前缀评分。

官方fine attention取决于training flag，单query RoPE也有广播位置语义；直接在eval使用等长并行Q/K配因果mask不能复现逐前缀推理。因此本分支训练逐前缀调用原decode_s2，保持其原权重/位置逻辑，避免未来访问；没有修改vendor源码。只算未来10个位置以控制开销。

## Smoke

32历史训练、16验证、16测试；20步训练、batch8、LR1e-5；验证固定采样seed选择CE最低checkpoint，含step0。生成T0.6/top-p0.9/sample_count2，逐股票日期相同seed。不是既有正式sample_count10的完整比较。

运行约66.5秒；identity初始化的官方生成最大误差0；原参数hash不变；未来token扰动不改变更早token；15项相关测试通过。

|起点|初始验证CE|20步后选中验证CE|
|---|---:|---:|
|identity|3.167376|3.166624|
|旧收益Adapter|4.204184|4.157878|

token CE有所下降但不能当作收益效果改善。16测试路径的生成轨迹MAE：官方0.051305、identity训练后0.051705、旧Adapter迁移训练后0.068227。该样本/采样极小，仅示范链路，不支持有效性结论；没有依据部署替代原收益模型。旧Adapter接官方头的分布不匹配现象支持优先identity起点再训练，不能推断更多训练必然解决。

## 使用

脚本scripts/experiments/run_token_adapter_smoke.py自动比较两种起点，使用项目内已下载官方权重：

    .runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_token_adapter_smoke.py --output-dir experiments/csi300_aligned/models/token_adapter_smoke_new --steps 20 --sample-count 2

模型入口src/kronos_continual/token_adapter.py，decode_s1/decode_s2可直接交给官方KronosPredictor；恢复时按原权重snapshot构造TokenPredictionAdapter，再加载checkpoint['adapter']至model.adapter，然后eval。

本次输出experiments/csi300_aligned/models/token_adapter_smoke_20261105/：token_metrics.csv、training_history.csv、generation_metrics.csv、predictions.csv、checkpoints/、audit.json、run_config.json、source_manifest.json、selected_samples.csv、status.json。

后续先扩大独立train/val并对比官方零样本、identity token Adapter、旧权重迁移token Adapter、原收益Adapter；验证CE与实际生成收益MAE/RMSE/IC/RankIC分开。最终使用同样股票日期/采样预算/成本，不能用teacher-forced token准确率替代free-running生成预测效果。
