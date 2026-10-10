# CSI300 模型对照

本目录只放 CSI300 对齐任务下的模型结果，每个方法使用独立子目录：

- `zero_shot/`：Kronos 零样本预测；
- `linear_head/`：线性预测头；
- `prediction_adapter/`：预测 Adapter；
- `ranking_adapter/`：排序 Adapter；
- `adapter_continual/`：Adapter 加持续学习；
- `continual_abc/`：Continual ABC；
- `continual_abc_mae_gate_tol001/`：带 MAE 门控的 Continual ABC；
- `odestream_direct_24x1/`：独立 ODEStream，单变量 close、24→1，不接入 Kronos 模块；
- `odestream_direct_24x1_smoke/`：ODEStream 小规模连通性检查。

ODEStream 使用原始方法口径，不能与 90→10 的 Kronos、Adapter 和 Continual 结果视为完全同任务指标。
