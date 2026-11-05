# HCA 模型迁移包（2026-11-05）

主配置：**Prediction checkpoint + 在线 Adapter/原头联合更新 + rank-2 训练标签 PCA 残差**，即 `hca_pca_final.safetensors`。这是已保存 HCA 主配置的归档，不是独立验证选出的全局最优模型。论文对照和负向结果见 [实验总结](../../docs/hca_archive_20261105/EXPERIMENT_SUMMARY.md)。

## 保存了什么

|文件|用途|
|---|---|
|`hca_pca_final.safetensors`|HCA-PCA 主模型，Adapter、原头、系数头和固定期限基齐全|
|`prediction_static.safetensors`|原始离线 Prediction Adapter，历史时序回放的起点|
|`prediction_online_final.safetensors`|普通在线 A，无期限残差|
|`hca_random_final.safetensors`|随机期限基消融；Endpoint 收益高于 PCA，但 Path 收益较低|
|`hca_dct_final.safetensors`|DCT 前两频段对照|
|`hca_frozen_head_final.safetensors`|冻结原头对照；Path 收益较高、Endpoint 较低|
|`hca_pca_training_state.pt`|原始主模型训练状态，含优化器、RNG、已消费标签及版本位置|
|`prediction_static_original.pt`、`rankic_original.pt`|两类离线起点的原始字节副本，供旧脚本 SHA256 核验及恢复路径使用|
|`manifest.json`、`verification.json`|来源、SHA256、模型状态版本、逐参数一致性及历史预测检查|
|`basis_manifest.json`、`historical_run_config.json`|PCA 实际矩阵、拟合范围与原运行配置|

所有 `*_final` 在线权重均为 **2026-06-05 当天预测发布后、成熟标签更新后的状态**。主模型版本216；它供后续预测使用。历史226日的75.617%收益来自每日不同版本的先预测后更新流，不能用最终权重回测整段历史来复现。最后一天历史预测使用前一日版本215。

## 新电脑恢复

```powershell
git clone --branch codex/a-online-adaptation-icde https://github.com/Zane-wzn/kronos-stock-forecast.git
cd kronos-stock-forecast
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r models/hca_best/requirements.txt
```

`.safetensors` 为普通 Git 内的小文件；原始 `.pt` 由已有 Git LFS 规则保存。`git lfs pull` 必须成功后才能把 `.pt` 当作模型加载。只需要推理时优先用 safetensors。

### 从已提取 hidden 预测

```powershell
python scripts/experiments/predict_hca_archive.py --hidden path/to/hidden.npy --output path/to/predictions.csv
```

输入 `[N,512]` float32 的 **冻结 Kronos 最后一位置 hidden、尚未经过 Adapter**。输出未来1—10交易日累计收益、十步均值 Path 信号、最后一步 Endpoint 信号。输出数值是收益比例，不是股价。

直接调用：

```python
from src.kronos_continual.hca_portable import load_model
model = load_model('models/hca_best/hca_pca_final.safetensors')
# model.eval() 已设置；hidden 是 torch.float32 的 [N,512] 张量。
with torch.inference_mode():
    cumulative_returns = model(hidden)
```

### 从90日价量 CSV 预测

```powershell
python scripts/experiments/predict_hca_csv.py --csv path/to/stock.csv --output path/to/forecast.json
```

CSV 至少有 `date,open,high,low,close,volume`；可有 `amount`。按日期排序取最近90条，必须是已经收盘且可用的交易日。首次运行下载官方 Kronos/tokenizer；支持 `--model-name`、`--tokenizer-name` 指向已有本地权重目录。

官方依赖为 `NeoQuasar/Kronos-small` 和 `NeoQuasar/Kronos-Tokenizer-base`。本机最近实验记录的 snapshot 分别是 `901c26c1332695a2a8f243eb2f37243a37bea320`、`0e0117387f39004a9016484a186a908917e22426`；脚本按这些 revision 下载。原226日 hidden 提取没有记录 revision，不能仅凭本机 snapshot 认定它们必然等于历史提取版本。原测试 hidden SHA256 已保存并核验。

历史预处理见 `scripts/data/build_continual_csi300_hidden_cache.py`：六字段 OHLCV/amount 各自在90日输入内做 z-score（std+1e-5），截断[-5,5]；时间特征顺序 minute/hour/weekday/day/month。缺少 amount 时用 OHLC 均值×volume 代理。冻结 tokenizer 编码后取 Kronos `decode_s1` 的 `context[:,-1,:]`。

### 恢复旧运行器所需初始权重

```powershell
Copy-Item models/hca_best/prediction_static_original.pt experiments/csi300_aligned/models/prediction_adapter/model.pt
New-Item -ItemType Directory -Force .runtime | Out-Null
Copy-Item models/hca_best/rankic_original.pt .runtime/D_model_rankic_pilot_original.pt
```

原运行器还依赖原始行情、`meta.csv`、`y.npy`、完整 hidden 缓存、参考预测与历史训练调度。迁移包没有收纳这些大规模逐样本文件；它是模型与论文证据归档，**不是完整数据备份**。部分旧实验目录只有目录或不完整副本，不能从目录名推断结果齐全。训练状态包含优化器/RNG，但旧引擎恢复还需匹配的 journals/prediction_shards，不是仅提供一个 `.pt` 即可原地续跑。

## 验证与解释边界

六份推理权重均逐张量与原 checkpoint 完全相等。用真实最后交易日297条样本、前一日状态复算已发布预测，最大绝对误差约1e-16，详见 `verification.json`。另用真实股票CSV和本地官方权重通过完整90日价量→hidden→HCA收益预测链路；17项Attention/LoRA/Token及在线引擎相关测试通过。原HCA内置成熟标签、未来标签置毒、断点恢复等正确性检查通过。尚未在另一台电脑重新下载官方权重或重建完整历史数据。

全窗口只有一个市场与主种子100，已多轮查看，固定成分股面板保留幸存者/成分前视偏差。HCA 相对普通在线 A 的 Path 收益差为+3.260个百分点，95%区间跨零，不能写成稳定显著优越。
