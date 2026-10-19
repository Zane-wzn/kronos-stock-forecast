"""Paired Path/Endpoint evaluation; realized next-day portfolio returns only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate.metrics import EvaluationConfig, evaluate_frame, clean_json
from src.doubleadapt.compare import indices, return_stats
from .data import KEYS, PCOLS, ACOLS, atomic_csv, atomic_json


def paired_daily(candidate, frozen, draws, mode):
    if not candidate.as_of_date.equals(frozen.as_of_date):
        raise AssertionError("paired evaluation dates differ")
    rows = []
    for metric in ["ic", "rankic", "turnover"]:
        delta = candidate[metric].to_numpy()-frozen[metric].to_numpy()
        boot = delta[draws].mean(1)
        rows.append({"signal": mode, "metric": metric, "delta": float(delta.mean()),
                     "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
    c, f = candidate.top_net.to_numpy(), frozen.top_net.to_numpy()
    cs, fs = return_stats(c), return_stats(f)
    cb, fb = return_stats(c[draws]), return_stats(f[draws])
    for metric in cs:
        boot = cb[metric]-fb[metric]
        rows.append({"signal": mode, "metric": metric, "delta": float(cs[metric]-fs[metric]),
                     "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
    return rows


def evaluate_pair(frozen, mars, out):
    out = Path(out)
    frozen = frozen.sort_values(KEYS).reset_index(drop=True)
    mars = mars.sort_values(KEYS).reset_index(drop=True)
    if not frozen[KEYS].equals(mars[KEYS]) or frozen.duplicated(KEYS).any():
        raise AssertionError("missing/duplicate paired keys; intersections prohibited")
    if not np.array_equal(frozen[ACOLS].to_numpy(), mars[ACOLS].to_numpy()):
        raise AssertionError("paired target conflict")
    for frame in [frozen, mars]:
        if not np.isfinite(frame[PCOLS+ACOLS].to_numpy()).all():
            raise FloatingPointError("nonfinite evaluation values")
        if not np.allclose(frame.pred_signal, frame[PCOLS].mean(1), rtol=0, atol=1e-7) or not np.allclose(frame.pred_endpoint_return, frame[PCOLS[-1]], rtol=0, atol=1e-7):
            raise AssertionError("signal semantics changed")
    config = EvaluationConfig(dataset="CSI300", top_k=30, drop_n=3, min_holding_days=5, cost_rate=.0015,
                              bootstrap_reps=5000, bootstrap_block=10, seed=20261013)
    daily, rows = {}, []
    for name, frame in [("frozen", frozen), ("mars_cl", mars)]:
        result = evaluate_frame(frame, config)
        # Re-evaluate the serialized CSV through the same public interface.
        again = evaluate_frame(pd.read_csv(out/name/"predictions.csv", dtype={"stock": str}), config)
        first = json.dumps(clean_json(result), sort_keys=True, allow_nan=False)
        second = json.dumps(clean_json(again), sort_keys=True, allow_nan=False)
        if first != second:
            # CSV encoding can introduce <1e-15 numeric changes. Repeat the same
            # file twice, the exact reproducibility criterion in the protocol.
            repeated = evaluate_frame(pd.read_csv(out/name/"predictions.csv", dtype={"stock": str}), config)
            if second != json.dumps(clean_json(repeated), sort_keys=True, allow_nan=False):
                raise AssertionError("same prediction file evaluated inconsistently")
            result = again
        atomic_json(out/name/"evaluation.json", clean_json(result))
        error = frame[PCOLS].to_numpy()-frame[ACOLS].to_numpy()
        for mode in ["path", "endpoint"]:
            cs = pd.DataFrame(result["daily"][mode]["cross_section"])
            port = pd.DataFrame(result["daily"][mode]["portfolio"])
            if not np.allclose(port.transaction_cost, .0015*port.turnover, rtol=0, atol=1e-14) or not np.allclose(port.top_net, port.top_raw-port.transaction_cost, rtol=0, atol=1e-14):
                raise AssertionError("portfolio cost semantics mismatch")
            merged = cs.merge(port[["as_of_date", "top_net", "turnover", "transaction_cost"]], on="as_of_date", validate="one_to_one").sort_values("as_of_date").reset_index(drop=True)
            if len(merged) != 226 or not np.isfinite(merged.drop(columns="as_of_date").to_numpy()).all():
                raise AssertionError("incomplete/nonfinite daily evaluation")
            daily[(name, mode)] = merged
            atomic_csv(out/name/f"{mode}_daily.csv", merged)
            atomic_csv(out/name/f"{mode}_portfolio.csv", port)
            monthly = merged.assign(month=merged.as_of_date.str[:7]).groupby("month").agg(
                ic=("ic", "mean"), rankic=("rankic", "mean"), turnover=("turnover", "mean"),
                net_return=("top_net", lambda v: (1+v).prod()-1)).reset_index()
            atomic_csv(out/name/f"{mode}_monthly.csv", monthly)
            metrics = result["metrics"][mode]
            rows.append({"branch": name, "signal": mode, **return_stats(merged.top_net.to_numpy()),
                         "ic": merged.ic.mean(), "rankic": merged.rankic.mean(), "turnover": merged.turnover.mean(),
                         "trajectory_mae": np.abs(error).mean(), "trajectory_rmse": np.sqrt(np.square(error).mean()),
                         "direction_accuracy": metrics["direction"]["accuracy"]})
    draws = indices(226, 5000, 10, 20261013)
    paired = []
    for mode in ["path", "endpoint"]:
        paired += paired_daily(daily[("mars_cl", mode)], daily[("frozen", mode)], draws, mode)
    atomic_csv(out/"metrics.csv", pd.DataFrame(rows))
    atomic_csv(out/"paired_bootstrap.csv", pd.DataFrame(paired))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, mode in zip(axes, ["path", "endpoint"]):
        for name in ["frozen", "mars_cl"]:
            d = daily[(name, mode)]
            ax.plot(pd.to_datetime(d.as_of_date), (np.cumprod(1+d.top_net)-1)*100, label=name)
        ax.set_ylabel(f"{mode} net return (%)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"returns.png", dpi=160)
    plt.close(fig)
    updates = pd.read_csv(out/"update_log.csv")
    lines = ["# 页面 C + MARS-CL 配对实验", "", "固定协议的回顾性在线比较；测试期此前已经查看。",
             "使用页面 C Prediction Adapter checkpoint（历史页面净累计收益62.48%）；原 Kronos、C Adapter 和 C Linear 全冻结，只有新增残差适配器更新。",
             "只读取成熟标签；固定历史训练样本用于梯度，最近10个完整成熟测试日期用于 shadow 门控。shadow 并非训练期历史验证集。",
             "相同checkpoint的 C-MARS-CL 与 C-Static 差异衡量持续学习模块整体增量；不把 D-RankIC 或独立GRU差异归因于本模块。",
             "论文中方法名仅为工作名称，未进行新颖性检索，不声称已证明原创性。",
             "", "|分支|信号|净累计收益|Sharpe|最大回撤|IC|RankIC|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['branch']}|{r['signal']}|{r['net_cumulative_return']:.2%}|{r['sharpe']:.3f}|{r['max_drawdown']:.2%}|{r['ic']:.5f}|{r['rankic']:.5f}|")
    lines += ["", f"更新次数 {len(updates)}；接受率 {updates.accepted.mean():.2%}。逐次成熟数量、排序变化、残差幅度见 update_log.csv 和 shadow_validation.csv。",
              "Top30/Drop3，最低持有5日，单向15bps，持仓收益仅 actual_return_1；统一 evaluate 接口的换手/成本实现原样沿用。",
              "Path 与 Endpoint 独立组合；共享非循环日期块 block=10、5000次、统计seed=20261013。收益/Sharpe/回撤区间重采样已实现日收益。",
              "单seed结果不足以声称可靠增益。三个seed复用同一个基础模型，衡量新增残差模块随机性，不衡量Kronos预训练种子随机性。",
              "", "![净值](returns.png)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows
