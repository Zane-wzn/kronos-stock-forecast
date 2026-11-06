# HCA 论文实验与模型迁移总结

整理日期：2026-11-05。依据本地已保存配置、预测重评结果与实验报告整理，未重新训练。原始紧凑证据副本在 [evidence/](evidence/)，逐文件来源与 SHA256 在 [evidence_manifest.json](evidence_manifest.json)。权重及恢复入口见 [模型迁移包](../../models/hca_best/README.md)。

后续补充：[Chronos-2、TimesFM 3.0、ODEStream完整预测恢复与统一重评](../baseline_comparison_20261105/REPORT.md)。Chronos/TimesFM完成结果已找回并按HCA的Top30规则重算；ODEStream查明共享模型同目标日逐股读取标签更新，原结果仅作历史诊断，不能作为时序公平的确认性基线。

## 当前保留版本与“最优”的口径

保留 **Prediction 起点的 HCA-PCA** 为主配置：冻结 Kronos，512→64→512 残差 Adapter，基础512→10收益头，附加512→2系数头和固定10×2 PCA基；在线联合更新 Adapter、原头、系数头。PCA仅用525,700条训练标签，rank2解释90.63%的训练目标方差；这不是测试误差解释率。初始系数为零，接入时预测不变。

没有一个版本同时在收益、误差、排序、回撤和两个信号上占优。主版本是保留原论文主线的工程选择，不能称经独立验证证明的全局最优。随机基 Endpoint 收益更高、冻结原头 Path 收益更高，但各有反向指标；这两份权重也保留，供后续重审而非事后替换主结论。

## 同口径完整226日主链条

CSI300固定面板；2025-07-01—2026-06-05；66,103股票日期键；90日输入、未来10交易日累计收益。Path是十步累计收益的均值，Endpoint是第10步累计收益。回测Top30/Drop3、最少持有5日、单边15bps，`actual_return_1`兑现次日收盘收益，252日年化。以下均为净累计收益，不是年化收益。

|阶段|轨迹RMSE↓|Path IC|Path RankIC|Path净收益|Endpoint净收益|Path最大回撤|
|---|---:|---:|---:|---:|---:|---:|
|Kronos零样本|0.102351|-0.063652|-0.037236|21.603%|3.169%|-10.737%|
|冻结特征＋线性头|0.070255|-0.044550|-0.027428|6.393%|16.864%|-16.641%|
|静态Prediction Adapter|0.064830|0.051746|0.046458|62.477%|54.091%|-8.475%|
|普通在线A|0.064246|0.069215|0.057865|72.357%|80.289%|-11.767%|
|HCA-PCA（主配置）|0.064335|0.069050|0.058550|75.617%|85.115%|-12.519%|
|HCA-Random|0.064270|0.069047|0.057871|71.577%|86.575%|-12.021%|
|HCA-DCT|0.064361|0.068584|0.058304|74.173%|73.421%|-10.990%|

来源：[逐层对照](evidence/prediction_progression_20261102/metrics.csv)、[原HCA完整比较](evidence/hca_online_full_20261020/comparison_metrics.csv)、[DCT比较](evidence/prediction_dct_full_20261102/metrics.csv)。RMSE单位为累计收益比例。

静态Adapter→在线A：Path收益约+9.881个百分点、Endpoint约+26.197个百分点，RMSE下降；Path Sharpe从2.859降到2.609，回撤扩大。A→PCA：Path+3.260个百分点、Endpoint+4.826个百分点，但RMSE略升、IC略降、Path回撤扩大。PCA−A的Path收益差95%区间为[-8.577,+20.412]个百分点，未证明稳定显著增益。

零样本→监督Adapter包含输出形式、训练目标和选模等整体变化，不能全部归因于Adapter结构。线性头对照接近结构消融，但历史记录未证明目标/预算/选模严格相同。不要将旧页面另一股票池或225日窗口的数字混入此表。

## 为论文做过的其他对比与消融

|实验|比较目的与设置|保存结果与可用结论|
|---|---|---|
|Prediction/RankIC两个离线起点|分别比较Static、A、PCA、Random；每流226日、216次更新、69,120标量成熟标签|Prediction/PCA Path75.617%；RankIC/PCA40.500%，低于其A的41.333%。HCA优势随起点改变，不能把两起点当作多种子泛化。|
|HCA增益归因，20条流|2起点×Adapter可训练/冻结×A/HCA/冻结原头/匹配更新范数/方向投影|HCA有效头更新范数约为A的1.395—1.408倍，主要放大PCA内方向；四组HCA−A Path收益差区间均跨零。匹配范数与HCA排序极近；不证明唯一收益来源。|
|冻结原头HCA|冻结基础收益头，只更新Adapter与系数头|Prediction/Live Path79.920%高于主HCA，Endpoint66.960%低于主HCA；保留为指标冲突对照，不据测试收益事后宣称全面最优。|
|DCT替换PCA，完整联合在线|同一Prediction起点、10×2基、AdamW/反馈/216步一致|PCA Path75.617%/Endpoint85.115%；DCT74.173%/73.421%。本次DCT没有超过PCA；不涵盖其他秩/选频。|
|CFUS效用子空间|Frozen、目标PCA、残差PCA、DCT、Random、CFUS、Full；表示/原头冻结，SGD；选择期定步长，后86日评价|Prediction连续CFUS相对Frozen MSE改善0.1834%；RankIC相对Full恶化1.7526%。未通过继续条件；不能主张稳定优于调优Full/PCA。|
|谱/频率机制|DCT2/4、高频负对照、软收缩、全维SGD；同86日误差评价|RankIC DCT2/4比PCA MSE下降约1.64%/1.68%，仍略差于Full；Prediction主要候选选择零步长。低频误差集中可能来自累计目标本身，不证明低频误差可预测。|
|共同/个股残差分离|在普通在线A之外加共同、个股、双头、未分离校准；按选择期定步长|共同、个股和未分离分支均选0；后86日与A完全相同，没有被接纳的增益。该86日数字不能与226日直接排名。|
|Temporal Attention扩大CPU实验|60股、80日测试、3种子；Adapter/等权汇总/学习attention各做静态与在线|在线attention RMSE仅改善约0.042%，MAE恶化约0.104%，Path RankIC0.014818→0.013492。Endpoint收益几乎被等权汇总复现，不支持attention独立稳定增益。|
|内部LoRA×目标四组|末两层Q/V rank4；原目标/标准化+排序目标；60股、3种子、两80日块|12个组种子均按预设验证门回退epoch0，不能把回退后的相同测试收益当作LoRA增益。一次稀疏小面板筛选，不证明LoRA普遍无效。|
|Token Adapter|冻结官方tokenizer/Kronos/粗细head；逐位置Adapter；identity与旧收益Adapter迁移起点|32训练/16验证/16测试smoke。CE略降；生成轨迹MAE官方0.051305、identity训练后0.051705、迁移0.068227。仅链路验证，未支持替换收益模型。|
|MARS-CL复核|冻结Page1-C，新增有界残差；本地仅seed100/102可完整配对|Path静态→MARS：seed10062.4766%→59.0256%，seed10267.9646%→67.5903%；RankIC下降、MAE略降。seed101副本不完整，排除。未核验缺失的原训练日志。|
|旧Continual/动态Hedge|D-RankIC，月末成熟标签更新、Replay、静态与持续分支动态融合|版本文档Path静态39.06%、持续33.70%、Hedge50.72%；是另一更新策略/起点，不把其改善解释为HCA增益。|
|ODEStream|close单变量24→1独立流式模型|旧225日公共交集摘要净收益27.299%；输入/标签/公共样本窗口与HCA不同，仅作清楚标注协议的外部方法对照。|
|DoubleAdapt|2026-11-06补查种子子目录，恢复GRU 90→10三个完整226日种子|同口径重评Path净收益56.122%/19.817%/26.275%，均值34.071%；原运行配置仍缺失。预测、任务索引、检查点与源码已归档，见[恢复报告](../paper_external_baselines_20261106/README.md)。|

对应证据：[归因报告](evidence/hca_gain_attribution_20261028/REPORT.md)、[CFUS](evidence/cfus_round1_20261022/REPORT.md)、[谱验证](evidence/spectral_validation_20261028/REPORT.md)、[DCT](evidence/prediction_dct_full_20261102/REPORT.md)、[残差分离](evidence/prediction_split_residual_20261103/REPORT.md)、[Attention](evidence/temporal_attention_pilot_60x80_20261104/REPORT.md)、[LoRA](evidence/lora_factorial_cpu_20261105/REPORT.md)、[Token](evidence/token_adapter_smoke_20261105/REPORT.md)、[MARS复核](evidence/mars_cl_result_check_20261028/RESULT_CHECK.md)、[Hedge版本](../VERSION_1.0_ADAPTER_CONTINUAL_HEDGE.md)、[ODEStream摘要](../../results/cumulative_return_curves/odestream_vs_existing/odestream_vs_existing_summary.csv)。

## 论文可以保留的机制描述

HCA用固定期限基组织新增残差的更新方向；完整标签成熟后更新，预测先发布、更新次日生效。基于当前实现，线性系数支路可以合并进线性基础头，函数类没有扩大；只有新增支路是rank2，整体预测头及总更新不被限制在rank2。零初始化证明初始函数保持，不能推出后续稳定、抗过拟合或收益不降。

当前主要证据支持研究“期限结构如何改变在线优化”，不支持“短期部分标签提前改善长期预测”：这些主实验等完整10步标签成熟才更新。训练目标PCA解释率高也可能由累计收益的期限重叠结构产生，不能直接当作适配残差或方法必要性证据。

## 写作时必须保留的限制

- 主226日实验是单市场、主种子100、已反复查看窗口；其他小面板的3种子不能替代主模型多种子确认。
- 固定成分股及未来完整窗口可评价样本筛选保留成分前视/幸存者偏差；收盘收益代理不包含完整停牌、涨跌停和流动性执行约束。
- 日期块bootstrap重采样已实现损失/收益，不重训练、不重建持仓；未校正多重比较。
- MatchedStep使用同日HCA更新幅度，是oracle机制诊断，不是部署可用策略。
- 59/50/86日机制实验、60股80日实验、16样本smoke与226日全股票池实验分别解释。
- 原始行情、全量预测/hidden以及部分旧运行产物没有包含在本次Git归档。恢复推理与重建完整实验是两项不同完成标准。

建议论文叙事：**离线任务适配 → 完整成熟反馈驱动的在线适应 → 期限结构引导的残差参数化 → 正负结果和适用边界**。独立新窗口/市场、主模型多种子和完整成本实验仍是未来工作。
