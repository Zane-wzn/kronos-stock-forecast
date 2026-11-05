# HCA：多步预测中基于期限耦合残差的在线适配

方法名称：Horizon-Coupled Adapter（HCA）。本文给出与当前实现一致的方法定义，包括模型结构、初始化、训练期基构造、延迟监督、优化和部署时序。第 1—8 节可组织为论文 Method，第 9—11 节用于实现细节和实验设计。整理日期：2026-10-22。

## 1. 问题定义与总体思路

对序列 $i$，在预测起点 $o$ 观测长度为 $L$ 的历史窗口 $X_{i,o}\in\mathbb R^{L\times p}$，并一次性发布未来 $H$ 步预测 $\hat{\mathbf y}_{i,o}\in\mathbb R^H$。金融实例使用累计收益目标：

$$
y_{i,o,k}=\frac{P_{i,o+k}}{P_{i,o}}-1,\qquad k=1,\ldots,H.
$$

其中 $o+k$ 按交易日历计数。当前输入长度 $L=90$、输出长度 $H=10$；一般方法要求骨干产生固定维度的历史表示，输出为 $H$ 维实值向量。

模型部署后持续面临分布变化，而用于适配的完整预测路径只能在未来逐步观测。HCA 保留预训练表示，以训练期确定的固定期限基组织输出残差，让多个预测步共享少量可更新的残差系数，并以完整成熟标签驱动因果在线适配。

完整框架包括三个衔接部分：

1. 冻结骨干、可训练共享 Adapter 和基础多步头，提供可适配预测器。
2. 在线冻结的期限基与零初始化系数头，构成期限耦合残差支路。
3. 先预测后更新的成熟标签协议，规定反馈何时可用以及新版本何时生效。

该定义对应当前完整成熟版本 HCA-PCA。随机期限基用于结构消融，不属于另一个主模型。

## 2. 冻结骨干与共享特征适配

冻结骨干 $F_\Theta$ 的参数 $\Theta$，提取历史表示：

$$
h_{i,o}=F_\Theta(X_{i,o})\in\mathbb R^d.
$$

当前使用 Kronos，$d=512$。在隐藏表示上使用残差 Adapter：

$$
\begin{aligned}
z_{i,o}^{(t)}&=A_{\phi_t}(h_{i,o}),\\
A_\phi(h)&=h+W_u\operatorname{GELU}
\bigl(W_d\operatorname{LN}_{\gamma,\beta_{\rm LN}}(h)+b_d\bigr)+b_u.
\end{aligned}
$$

其中 $W_d\in\mathbb R^{q\times d}$、$W_u\in\mathbb R^{d\times q}$，当前 $q=64$；$\phi$ 包括归一化与两层线性映射参数。代码在 GELU 后保留 Dropout 模块，但整个在线阶段保持 eval 模式，因此在线预测和梯度更新均关闭 Dropout。

原基础多步头为：

$$
g_{\psi_t}(z)=W_tz+b_t,\qquad
\psi_t=(W_t,b_t),\quad W_t\in\mathbb R^{H\times d}.
$$

初始化的 $\phi_0,\psi_0$ 从已训练的静态 checkpoint 继承。本文机制从该 checkpoint 出发，不重新定义离线训练配方。当前已有 Prediction 与 RankIC 两类起点；论文可从 Prediction 起点展开，其他起点用于迁移与敏感性实验，并分别交代离线目标。

由于骨干冻结，可以缓存 $h_{i,o}$。在线更新时必须用当前 $A_{\phi_t}$ 重新计算 $z_{i,o}^{(t)}$；Adapter 更新后，旧 $z$ 不能当作当前可训练表示复用。当前代码从缓存 hidden 重新运行当前 Adapter。

## 3. 训练期构造固定期限基

HCA 用固定矩阵 $B\in\mathbb R^{H\times r}$ 表示残差支路的共享期限方向，$r<H$。当前版本对训练 split 的完整目标路径做 PCA。

设训练目标为 $\{\mathbf y_n\}_{n=1}^N$，计算：

$$
\mu=\frac1N\sum_{n=1}^N\mathbf y_n,\qquad
\Sigma=\frac1N\sum_{n=1}^N
(\mathbf y_n-\mu)(\mathbf y_n-\mu)^\top.
$$

按特征值降序选择前 $r$ 个特征向量：

$$
\Sigma v_j=\lambda_jv_j,\quad
\lambda_1\ge\cdots\ge\lambda_H,\qquad
B=[v_1,\ldots,v_r],\quad B^\top B=I_r.
$$

为便于复算，每列绝对值最大元素的符号固定为正，再归一化为单位范数。浮点实现中的正交性按容差检查。重复特征值可能引起不同求解器返回不同坐标，因此同时保存实际矩阵及 hash。

当前 $H=10,r=2$，使用 525,700 条训练路径拟合，训练标签最晚日期为 2024-12-27，早于在线测试起点。保存的前两维解释方差为 90.63%；它描述训练目标协方差，不表示测试误差或适配残差的解释比例。

$B$ 在部署后冻结。均值 $\mu$ 只用于协方差估计，不在残差输出中额外加回。当前 $r=2$ 为固定配置；未来选秩应在验证期完成。

PCA 估计的是目标路径变化方向，并非预测误差方向或因果关系。论文应将“该方向有利于在线修正”表述为方法动机及实验假设。

## 4. 期限耦合残差头与最终输出

系数头将当前表示 $z$ 映射为 $r$ 个残差系数：

$$
\alpha_{\omega_t}(z)=C_tz+c_t,\qquad
\omega_t=(C_t,c_t),\quad C_t\in\mathbb R^{r\times d}.
$$

经固定期限基重建完整残差路径：

$$
\delta_{i,o}^{(t)}=B\alpha_{\omega_t}(z_{i,o}^{(t)})\in\mathbb R^H.
$$

HCA 的完整预测函数为：

$$
\boxed{
f_t(X_{i,o})
=W_tA_{\phi_t}(F_\Theta(X_{i,o}))+b_t
+B\left[C_tA_{\phi_t}(F_\Theta(X_{i,o}))+c_t\right].
}
$$

两个输出支路共享 $z$，残差支路额外使用固定 $B$。在代码的行向量批量约定下，等价写法为：

$$
\hat Y=ZW_t^\top+\mathbf1 b_t^\top+
(ZC_t^\top+\mathbf1 c_t^\top)B^\top.
$$

~~~mermaid
flowchart LR
    X["历史窗口 X"] --> F["冻结 Kronos"]
    F --> H["hidden h：512 维，可缓存"]
    H --> A["可训练共享 Adapter"]
    A --> Z["当前表示 z"]
    Z --> G["可训练基础头 Wz+b"]
    Z --> C["零初始化系数头 Cz+c：r 维"]
    C --> B["固定期限基 B：H × r"]
    G --> ADD["相加，得到 H 步预测"]
    B --> ADD
    ADD --> LOG["发布并保存当日预测"]
    LOG --> UPDATE["完整标签成熟后，更新 Adapter 与两个输出头"]
~~~

### 4.1 初始函数保持

添加残差支路时令 $C_0=0,c_0=0$，于是对任意输入：

$$
f_0(X)=g_{\psi_0}(A_{\phi_0}(F_\Theta(X))).
$$

HCA 接入时保持原 checkpoint 的预测。该性质针对初始化；在线更新后，基础头、Adapter 和系数头均可变化。零初始化本身不提供更新幅度、预测误差或收益保证。

## 5. 跨 horizon 的反馈传播

耦合发生在残差支路的系数空间。对同一历史样本，令 $e_k=\partial\mathcal L/\partial\hat y_k$，未参与本次监督的坐标令 $e_k=0$。固定当前表示 $z$：

$$
\frac{\partial\mathcal L}{\partial\alpha}=B^\top e,\qquad
\nabla_C\mathcal L=(B^\top e)z^\top,\qquad
\nabla_c\mathcal L=B^\top e.
$$

各 horizon 的误差信号经 $B$ 的对应行进入共享的 $r$ 维空间。系数头更新后，残差的全部 $H$ 个坐标都可能变化，因此该支路提供显式期限耦合。

可以给出一个可推导的机制解释：固定 Adapter、基础头和 $z$，只对系数头做一次学习率 $\eta$、无裁剪的普通 SGD，则同一 $z$ 上的残差变化为：

$$
\Delta\delta=-\eta(\|z\|_2^2+1)BB^\top e.
$$

若只有第 $k$ 步参与监督，第 $j$ 步变化为：

$$
\Delta\delta_j=-\eta(\|z\|_2^2+1)
\langle B_{j,:},B_{k,:}\rangle e_k.
$$

期限基行向量的内积决定该简化更新中的传播强度和符号。正、负或零内积可以分别产生同向、反向或零传播；不能由“耦合”推出“其他 horizon 误差必然下降”。

对另一个固定表示 $z'$，同一次系数更新引起：

$$
\Delta\delta(z')=-\eta(z'^\top z+1)BB^\top e.
$$

这说明该简化情形中存在两个传播因素：特征内积决定历史样本更新如何影响另一个输入，$BB^\top$ 决定 horizon 间如何耦合。该推导只说明单次参数更新的作用路径，不保证分布变化时历史反馈仍有利于未来样本。

当前算法使用 AdamW、联合梯度裁剪并同时训练 Adapter 和基础头，上式解释残差支路的耦合来源，不是完整实际优化轨迹的精确公式。当前正式实验使用全部完整成熟 horizon，验证的是完整标签下的耦合适配；短期标签提前帮助长期预测需要另行实验。

### 5.1 与基础多步头的关系

基础模型通过共享 Adapter 已有跨 horizon 表示共享。HCA 增加固定基残差更新路径。因为 $W,C$ 同时可训练：

$$
Wz+b+B(Cz+c)=(W+BC)z+(b+Bc).
$$

整个输出函数可合并为普通仿射头。受 $r$ 维子空间约束的是新增残差 $\delta$，不是总输出，也不是全部在线修正。论文贡献应落在结构化参数化和在线更新机制，而不是表达能力增加。

## 6. 完整成熟约束与预算采样

对监督键 $e=(i,o,k)$，令 $a(i,o,k)$ 为标签实际可用日期，完整路径可用日期为：

$$
m(i,o)=\max_{1\le k\le H}a(i,o,k).
$$

最大值允许一般的非顺序到达；当前按序交易日数据中 $m(i,o)=a(i,o,H)$。令 $\mathcal U_t$ 为此前已消费的键，时刻 $t$ 的可选集合为：

$$
\mathcal A_t=\{(i,o,k):o<t,\ m(i,o)\le t,\ (i,o,k)\notin\mathcal U_t\}.
$$

固定协议选择 $\mathcal Q_t\subseteq\mathcal A_t$，并按 horizon 分解：

$$
\mathcal Q_{t,k}=\{(i,o):(i,o,k)\in\mathcal Q_t\},\qquad
|\mathcal Q_{t,k}|\le n_{\max}.
$$

当前可复现实例设置为：

- 按训练期股票可用性和固定 hash 预选 60 股；既有面板实际覆盖 59 股。
- 每日处理最新恰好完整成熟的测试 origin；每个 horizon 最多选择 32 股。
- 为沿用先前完整/前缀对照的预算，代码取完整成熟 origin 与该 horizon 对应前缀 origin 的股票交集，再按固定 hash 选股；实际训练输入和标签仍取完整成熟 origin。
- 前 $H=10$ 个测试日无完整成熟测试路径，此后每天最多一次更新。

股票交集是当前运行器的采样约定，不是 HCA 结构必需条件。采样不按测试误差、收益或标签数值筛选。HCA、A 和随机基消融使用相同实际监督键、日期与 optimizer step 数。

一次消费针对 $(i,o,k)$，不表示相邻起点目标互相独立。仅实际用于训练的 $\mathcal Q_t$ 加入消费集合；未选键不会被标记为已训练。当前调度不保证所有未选样本以后都会训练，也没有额外 replay。

## 7. 在线目标与优化

对历史样本，用当前参数重算：

$$
\tilde{\mathbf y}_{i,o}^{(t)}
=g_{\psi_t}(A_{\phi_t}(h_{i,o}))
+B\alpha_{\omega_t}(A_{\phi_t}(h_{i,o})).
$$

$\tilde{\mathbf y}^{(t)}$ 是训练时当前输出，区别于 origin $o$ 当时发布的 $\hat{\mathbf y}_{i,o}^{\rm issued}$；重算不会覆盖历史预测。

定义 $\mathcal K_t=\{k:|\mathcal Q_{t,k}|>0\}$，使用样本平均后 horizon 等权的目标：

$$
\mathcal L_t=
\frac1{|\mathcal K_t|}\sum_{k\in\mathcal K_t}
\frac1{|\mathcal Q_{t,k}|}
\sum_{(i,o)\in\mathcal Q_{t,k}}
\rho_\beta(\tilde y_{i,o,k}^{(t)}-y_{i,o,k}),
$$

$$
\rho_\beta(u)=
\begin{cases}
u^2/(2\beta),& |u|<\beta,\\
|u|-\beta/2,& |u|\ge\beta.
\end{cases}
$$

当前 $\beta=1$，收益以小数记录。所有 horizon 损失累积后反向传播一次，不对每个 horizon 单独更新。目标不包含排序、交易收益、一致性或残差幅度正则项。

在线参数为 $\theta_t=(\phi_t,\psi_t,\omega_t)$，$\Theta,B$ 冻结。计算 $g_t=\nabla_{\theta_t}\mathcal L_t$，对全部可训练参数联合裁剪：

$$
\bar g_t=g_t\min\left(1,\frac{1}{\|g_t\|_2}\right),
$$

零梯度时保持零。随后执行 AdamW，weight decay=0；Adapter 学习率 $10^{-5}$，基础头与系数头均为 $10^{-4}$。代码把两个输出头放在同一参数组，当前没有独立调节系数头学习率的执行路径。其余 AdamW 参数沿用 PyTorch 默认值：betas 为 $(0.9,0.999)$，eps 为 $10^{-8}$。

## 8. 因果部署时序与算法

令 $\theta_t$ 为日期 $t$ 发布预测时的参数。每日顺序：

1. 获取截至当日的输入，发布 $\hat{\mathbf y}^{\rm issued}_{i,t}=f_t(X_{i,t})$。
2. 无标签预测分片落盘，记录输入键、模型版本和 hash。
3. 检查完整成熟日期、选择 $\mathcal Q_t$，再访问选中且合法的标签。
4. 从历史冻结 hidden 重算当前表示与预测，计算损失并更新。
5. 标记实际训练键，保存模型、优化器、随机状态和进度；下一预测日启用新版本。

没有监督时 $\theta_{t+1}=\theta_t$。历史已发布预测不重写。完整成熟约束与先预测后更新是两个独立要求。

~~~text
Algorithm: HCA with complete-maturity feedback
Input: frozen F_Theta; pretrained phi_0, W_0, b_0;
       training paths; rank r; fixed sampler and budget

B <- leading r PCA vectors of training target covariance
C_0 <- 0; c_0 <- 0
U <- empty set
initialize AdamW for {phi, W, b, C, c}

for online date t in chronological order:
    h_t <- F_Theta(X_t) or verified frozen hidden cache
    z_t <- A_phi(h_t)
    issued_t <- W z_t + b + B(C z_t + c)
    persist(issued_t, current_version, prediction_hash)

    A_t <- complete-mature, previously unconsumed legal keys
    Q_t <- fixed_sampler(A_t, protocol, per_horizon_budget)
    if Q_t is not empty:
        read only labels indexed by Q_t
        recompute historical z <- A_phi(h) with current phi
        L <- sample means within horizons, then horizon mean
        compute gradients w.r.t. {phi, W, b, C, c}
        jointly clip global gradient norm to 1
        perform one AdamW step
        U <- U union Q_t
        increment version; save optimizer, model, RNG and progress
    activate resulting state for the next forecast date
~~~

## 9. 参数与复杂度

当前 Adapter 参数数为：

$$
2d+(qd+q)+(dq+d)=2dq+3d+q=67{,}136.
$$

基础头有 $H(d+1)=5,130$ 个参数，系数头额外增加 $r(d+1)=1,026$：

| 部件 | 当前维度 | 在线状态 | 参数数 |
|---|---|---|---:|
| Kronos | hidden $d=512$ | 冻结 | 不计入在线训练参数 |
| 共享 Adapter | $512\to64\to512$，含 LayerNorm | 更新 | 67,136 |
| 基础多步头 | $512\to10$ | 更新 | 5,130 |
| 系数头 | $512\to2$ | 更新，初始为零 | 1,026 |
| 期限基 | $10\times2$ | 冻结 buffer | 20 个常量，0 个训练参数 |
| HCA 合计 | — | — | 73,292 |

相对 A 的 72,266 个在线参数增加约 1.42%。每个样本增加 $O(rd+Hr)$ 支路计算，以及 $O(rd+r+Hr)$ 参数与固定基存储；AdamW 还维护新增可训练参数的一阶、二阶状态。这是增量复杂度，完整推理仍包含骨干、Adapter 和基础头。

直接 PCA 成本约 $O(NH^2+H^3)$。聚合一、二阶统计量可用 $O(H^2)$ 工作空间估计协方差；当前代码实际加载训练目标后计算，并未实现该流式内存优化。

缓存管理和输出日志不包含在新增参数预算里。当前没有所有文件存储的常数内存保证。效率实验应分别报告缓存适配与端到端耗时；监督键和 step 匹配不等同于计算量或墙钟时间匹配。

## 10. 论文机制表述与必要对照

建议使用的核心表述：

> HCA 将多步输出修正分解为基础预测支路与固定期限基上的低维残差支路。训练期 PCA 提供共享期限方向，零初始化系数头保持接入时的预测函数；在线成熟标签共同更新系数空间、基础头与共享 Adapter。多 horizon 反馈由此能够通过显式残差通道传播，同时遵守完整标签成熟和先预测后更新的因果约束。

英文方法概述：

> HCA augments a pretrained multi-step forecaster with a horizon-coupled residual branch. A horizon basis is estimated from the covariance of training target trajectories and frozen at deployment. An input-dependent coefficient head maps adapted backbone features to this basis, producing a residual trajectory added to the original prediction head. Zero initialization preserves the initial forecasting function. During deployment, the shared adapter, original head, and coefficient head are jointly updated using only fully matured target trajectories. Forecasts are committed before supervised updates, and the updated parameters are activated at the next forecast origin.

建议实验分别检验固定基、联合适配和监督时序：

| 对照 | 检验对象 | 解释边界 |
|---|---|---|
| Static | 是否需要在线更新 | 与各自初始 checkpoint 配对 |
| F：仅基础头更新 | 共享 Adapter 更新的作用 | 同监督键与采样预算 |
| A：Adapter + 基础头 | HCA 支路的增量价值 | 当前直接主对照 |
| HCA-Random | PCA 期限方向的作用 | 同秩、同额外参数、同初始函数 |
| $B=0$ | 去除有效残差的退化验证 | 与 A 一致，属于正确性检查 |
| 冻结 Adapter 和基础头，仅训系数头 | 隔离残差通道的传播 | 独立消融，不替代当前模型 |
| A 的学习率/更新幅度控制 | 是否仅为优化尺度变化 | 相同验证期选择规则 |
| 完整/逐步成熟 × A/HCA | 提前反馈下的额外耦合效应 | 逐步版本需要另实现和验证 |

普通全 horizon 独立残差头通常有 $H(d+1)$ 个参数，不能直接声称与秩 $r$ 的系数头参数相同。参数匹配控制需给出具体结构与实际数量；HCA-Random 已是当前同参数结构控制。

不应把普通头描述为完全不能传递跨期限信息，也不应声称总输出低秩。模型定义可以确定耦合通道存在，其对误差、稳定性和长期预测的效果依赖实验。

## 11. 当前实验实例与证据边界

CSI300 运行使用 $90\to10$，测试区间 2025-07-01—2026-06-05，226 日、66,103 个股票日期键。前 10 日不更新，随后 216 次更新，每次 320 个监督标量，共 69,120 个。最后一日更新不影响本窗口内已发布预测。

Path 为预测累计收益的 horizon 均值，Endpoint 为第 10 步；它们是同一路径的不同汇总。金融评测采用 Top30/Drop3、最短持有 5 日、单边 15bps 成本，以 actual_return_1 兑现日收益。轨迹与逐 horizon MAE/RMSE 则评价完整数值预测。

保存的 HCA-PCA 结果在 Prediction 起点上有收益改善与 IC 略降，在 RankIC 起点上没有一致改善。这些现象应在结果部分呈现；不改变上述模型定义，但限定性能主张。耦合机制存在不等于已证明普遍有效。

实现核对入口：

- scripts/experiments/run_delayed_feedback_pilot.py：Adapter 与基础模型。
- scripts/experiments/run_hca_online_full.py：PCA、固定基、零初始化头及运行审计。
- scripts/experiments/run_online_module_pilot.py：成熟访问、采样、当前表示重算与在线更新。
- experiments/csi300_aligned/models/hca_online_full_20261020/basis_manifest.json：实际基和训练边界。
- experiments/csi300_aligned/models/hca_online_full_20261020/run_config.json：超参数与预算。

本文只整理机制与论文表述，没有改动训练算法或已有结果。完整成熟版本与逐步成熟扩展应在文稿、代码和实验表中明确区分。

公式核验记录见同目录 HCA_MODEL_MECHANISM_CHECKS.json。使用保存的 PCA 基和固定随机种子的合成特征，系数梯度有限差分最大误差为 $5.18\times10^{-12}$ 以内，跨输入 SGD 残差传播与线性头合并均满足 $10^{-12}$ 误差阈值；参数数和监督预算已与保存配置交叉核对。这些是数学与配置核验，不代表新增模型性能实验。
