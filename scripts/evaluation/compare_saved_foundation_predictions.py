"""Recover saved TSFM/ODEStream forecasts and compare without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from evaluate.inputs import normalize_predictions
from evaluate.metrics import EvaluationConfig, clean_json, evaluate_frame
from src.doubleadapt.compare import indices, return_stats

KEYS = ["as_of_date", "stock"]
PC = [f"pred_return_{k}" for k in range(1, 11)]
AC = [f"actual_return_{k}" for k in range(1, 11)]
MODELS = ROOT / "experiments/csi300_aligned/models"
DEFAULT_OUT = ROOT / "docs/baseline_comparison_20261105"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_points(path):
    keep = set(KEYS + PC + AC + ["pred_signal", "actual_signal", "current_close",
                "pred_endpoint_return", "actual_endpoint_return", "label_dates", "pred_version",
                "input_start_date", "input_end_date", "model_version", "status"])
    keep.update(f"{prefix}_close_{k}" for prefix in ("pred", "actual") for k in range(1, 11))
    keep.update(f"future_date_{k}" for k in range(1, 11))
    keep.update(f"label_date_{k}" for k in range(1, 11))
    return normalize_predictions(pd.read_csv(path, usecols=lambda c: c in keep, dtype={"stock": str}, float_precision="round_trip"))


def source_entry(path):
    return {"path": str(path.relative_to(ROOT)).replace(chr(92), "/"),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def canonical(frame, ref, name):
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"Duplicate keys: {name}")
    aligned = ref[KEYS].merge(frame, on=KEYS, how="left", validate="one_to_one")
    if not np.isfinite(aligned[PC + AC].to_numpy()).all():
        raise AssertionError(f"Missing/nonfinite forecast or target: {name}")
    error = float(np.max(np.abs(aligned[AC].to_numpy() - ref[AC].to_numpy())))
    assert error < 1e-6, (name, error)
    aligned[AC] = ref[AC].to_numpy()
    for prefix, cols in (("pred", PC), ("actual", AC)):
        aligned[prefix + "_signal"] = aligned[cols].mean(axis=1)
        aligned[prefix + "_endpoint_return"] = aligned[cols[-1]]
    columns = KEYS + PC + AC + ["pred_signal", "actual_signal", "pred_endpoint_return", "actual_endpoint_return"]
    return aligned[columns], {"model": name, "source_rows": len(frame), "aligned_rows": len(aligned),
                               "extra_rows": len(frame) - len(aligned), "target_max_abs_difference_before_canonicalization": error}


def evaluate(name, frame, out, panel, signal_names=None):
    cfg = EvaluationConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=.0015,
                           annualization_days=252, bootstrap_reps=0)
    result = evaluate_frame(frame, cfg)
    save(out / "evaluations" / panel / f"{name}.json", result)
    rows, daily = [], {}
    for mode in ("path", "endpoint"):
        m = result["metrics"][mode]
        if m["point"]["status"] == "无法验证":
            continue
        port = pd.DataFrame(result["daily"][mode]["portfolio"])
        cross = pd.DataFrame(result["daily"][mode]["cross_section"])
        d = port.merge(cross, on="as_of_date", validate="one_to_one")
        actual_mode = signal_names.get(mode, mode) if signal_names else mode
        daily[actual_mode] = d
        d.to_csv(out / "evaluations" / panel / f"{name}_{actual_mode}_daily.csv", index=False)
        trajectory = result["metrics"]["trajectory_pooled"]
        rows.append({"panel": panel, "model": name, "signal": actual_mode, "rows": len(frame),
                     "dates": frame.as_of_date.nunique(), "start": frame.as_of_date.min(), "end": frame.as_of_date.max(),
                     "trajectory_mae": trajectory.get("mae"), "trajectory_rmse": trajectory.get("rmse"),
                     "mae": m["point"]["mae"], "rmse": m["point"]["rmse"],
                     "direction_accuracy": m["direction"]["accuracy"],
                     "balanced_accuracy": m["direction"]["balanced_accuracy"],
                     "ic": m["cross_section"]["ic"]["mean"],
                     "rankic": m["cross_section"]["rankic"]["mean"],
                     "turnover": m["portfolio"]["turnover"], **return_stats(d.top_net.to_numpy())})
    return rows, daily


def paired(daily, names):
    records = []
    for left, right in names:
        for signal in ("path", "endpoint"):
            a, b = daily[left, signal], daily[right, signal]
            assert a.as_of_date.equals(b.as_of_date)
            draws = indices(len(a), 5000, 10, 20261013)
            for metric in ("ic", "rankic", "net_cumulative_return"):
                if metric == "net_cumulative_return":
                    delta = return_stats(a.top_net.to_numpy())[metric] - return_stats(b.top_net.to_numpy())[metric]
                    samples = return_stats(a.top_net.to_numpy()[draws])[metric] - return_stats(b.top_net.to_numpy()[draws])[metric]
                else:
                    diff = a[metric].to_numpy() - b[metric].to_numpy()
                    delta, samples = diff.mean(), diff[draws].mean(1)
                low, high = np.quantile(samples, [.025, .975])
                records.append({"left": left, "right": right, "signal": signal, "metric": metric,
                                "delta": delta, "ci_low": low, "ci_high": high})
    return records


def load_ode(ref, out):
    path = MODELS / "odestream_direct_24x1/predictions.csv"
    raw = pd.read_csv(path, dtype={"stock": str}, float_precision="round_trip")
    assert not raw.duplicated(["date", "stock"]).any()
    assert np.array_equal(raw.update_order.to_numpy(), np.arange(len(raw)))
    assert raw[["date", "stock"]].equals(raw.sort_values(["date", "stock"])[["date", "stock"]].reset_index(drop=True))
    full = []
    for stock, g in raw.groupby("stock", sort=True):
        prices_path = ROOT / "data/csi300_daily/csi300_daily" / f"{stock.replace('.', '_')}.csv"
        prices = pd.read_csv(prices_path, usecols=["date", "close"]).dropna()
        prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
        prices = prices.loc[prices.close.gt(0)].sort_values("date").drop_duplicates("date")
        prices["as_of_date"] = prices.date.shift(1)
        prices["previous_close"] = prices.close.shift(1)
        joined = g.rename(columns={"date": "target_date"}).merge(
            prices[["date", "as_of_date", "close", "previous_close"]],
            left_on="target_date", right_on="date", how="left", validate="one_to_one")
        assert joined.as_of_date.notna().all()
        np.testing.assert_allclose(joined.true_close, joined.close, rtol=0, atol=1e-5)
        np.testing.assert_allclose(joined.current_close, joined.previous_close, rtol=0, atol=1e-5)
        joined["pred_return_1"] = joined.pred_close / joined.current_close - 1
        joined["actual_return_1"] = joined.true_close / joined.current_close - 1
        full.append(joined[KEYS + ["target_date", "pred_return_1", "actual_return_1", "update_order"]])
    frame = pd.concat(full, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    daily = raw.groupby("date", sort=True).agg(rows=("stock", "size"),
            first_update_order=("update_order", "min"), last_update_order=("update_order", "max"))
    daily["rows_preceded_by_same_target_day_label_update"] = daily.rows - 1
    daily.to_csv(out / "odestream_same_day_update_audit.csv")
    save(out / "odestream_audit.json", {
        "source": source_entry(path), "raw_rows": len(raw), "raw_target_dates": raw.date.nunique(),
        "target_start": raw.date.min(), "target_end": raw.date.max(),
        "stocks": raw.stock.nunique(), "same_day_later_predictions": int((daily.rows - 1).sum()),
        "fraction": float((len(raw) - len(daily)) / len(raw)),
        "ordering": "One shared model/optimizer; sorted target date then stock; predict_one then update_one for every row",
        "code": "scripts/experiments/run_odestream_csi300_24x1.py build_split, main streaming loop",
        "interpretation": "For previous-close cross-sectional publication, later stocks can use the current target-day labels of earlier stocks. Historical diagnostic only; not a causally matched baseline. No rerun performed.",
        "old_summary_rows": 66709, "recovered_raw_rows": len(raw),
        "note_on_old_rows": "66709 equals 67006 minus the 297 targets on 2025-07-01 after backdating to prior close outside comparison start; not a missing file discrepancy"})
    return frame


def plot_curves(out, daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kronos_zero_shot": "#8794a2", "Chronos2": "#7e57c2", "TimesFM3": "#dc8e25",
              "Prediction_static": "#2189a6", "Prediction_online": "#258a59", "HCA_PCA": "#c54853"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, signal in zip(axes, ("path", "endpoint")):
        for name, color in colors.items():
            d = daily[name, signal]
            nav = np.cumprod(1 + d.top_net.to_numpy()) - 1
            ax.plot(pd.to_datetime(d.as_of_date), nav * 100, label=name, color=color, lw=1.7)
        ax.axhline(0, color="#b6bcc3", lw=.7)
        ax.set_title(signal.capitalize() + " signal")
        ax.grid(alpha=.16)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Cumulative net return (%)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Saved forecasts on identical 66,103 stock/date keys (226 days)", fontsize=13)
    fig.text(.5, .01, "Top30 / Drop3 / minimum hold 5 / one-way cost 15bps. Retrospective fixed constituent panel.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(out / "cumulative_returns.png", dpi=160)
    fig.savefig(out / "cumulative_returns.svg")
    plt.close(fig)


def markdown_table(frame, signal, fields):
    d = frame.loc[frame.signal.eq(signal)]
    headers = ["模型"] + [x[1] for x in fields]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] + ["---:"] * len(fields)) + "|"]
    for row in d.to_dict("records"):
        values = [row["model"]]
        for key, _, style in fields:
            value = row[key]
            values.append("无法验证" if pd.isna(value) else format(value, style))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def main(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sources, audit, frames = {}, [], {}
    ref_path = MODELS / "prediction_adapter/predictions.csv"
    ref = read_points(ref_path).sort_values(KEYS).reset_index(drop=True)
    assert len(ref) == 66103 and ref.as_of_date.nunique() == 226
    paths = {"Chronos2": ROOT / "chronos2/results/predictions.csv",
             "TimesFM3": ROOT / "timesfm3/results/predictions.csv",
             "Prediction_static": ref_path,
             "Prediction_online": MODELS / "hca_online_full_20261020/streams/Prediction_A/predictions.csv",
             "HCA_PCA": MODELS / "hca_online_full_20261020/streams/Prediction_HCA_PCA/predictions.csv"}
    for name, path in paths.items():
        source = read_points(path)
        frames[name], a = canonical(source, ref, name)
        sources[name] = [source_entry(path)]
        audit.append(a)
    shards = sorted((MODELS / "zero_shot/predictions/by_date").glob("*.csv"))
    kronos = pd.concat([read_points(p) for p in shards], ignore_index=True)
    frames["Kronos_zero_shot"], a = canonical(kronos, ref, "Kronos_zero_shot")
    sources["Kronos_zero_shot"] = [source_entry(p) for p in shards]
    audit.append(a)
    # Preserve original bytes, including all quantiles, rather than only the summary.
    archive_dir = out / "prediction_archives"
    archive_dir.mkdir()
    for name, path in paths.items():
        with zipfile.ZipFile(archive_dir / f"{name}.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(path, "predictions.csv")
    with zipfile.ZipFile(archive_dir / "Kronos_zero_shot.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in shards:
            archive.write(path, "by_date/" + path.name)
    ode_path = MODELS / "odestream_direct_24x1/predictions.csv"
    with zipfile.ZipFile(archive_dir / "ODEStream_original.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(ode_path, "predictions.csv")
    print("Forecast archives created", flush=True)
    metadata_dir = out / "run_metadata"
    for folder in ("chronos2", "timesfm3"):
        for relative in ("config.json", "provenance.json", "requirements.lock.txt",
                         "results/summary.json", "results/run_identity.json",
                         "results/data_audit.json", "results/evaluation/comparison.csv"):
            p = ROOT / folder / relative
            if p.exists():
                target = metadata_dir / folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    for name in ("run_config.json", "summary.json"):
        target = metadata_dir / "ODEStream" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODELS / "odestream_direct_24x1" / name, target)
    point_dir = out / "normalized_points"
    point_dir.mkdir()
    for name, f in frames.items():
        f.to_csv(point_dir / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    rows, full_daily = [], {}
    for name in ("Kronos_zero_shot", "Chronos2", "TimesFM3", "Prediction_static", "Prediction_online", "HCA_PCA"):
        r, d = evaluate(name, frames[name], out, "full_226")
        rows.extend(r)
        full_daily.update({(name, mode): value for mode, value in d.items()})
        print("evaluated full panel " + name, flush=True)
    main_table = pd.DataFrame(rows)
    main_table.to_csv(out / "full_226_metrics.csv", index=False)
    pairs = paired(full_daily, [("HCA_PCA", "Chronos2"), ("HCA_PCA", "TimesFM3"),
                               ("Chronos2", "Kronos_zero_shot"), ("TimesFM3", "Kronos_zero_shot"),
                               ("HCA_PCA", "Prediction_online")])
    pd.DataFrame(pairs).to_csv(out / "paired_intervals_full_226.csv", index=False)
    # H1 is compared to H1; ODEStream is never expanded into a fake ten-step path.
    ode = load_ode(ref, out)
    common = ref[KEYS].merge(ode[KEYS], on=KEYS, how="inner", validate="one_to_one").sort_values(KEYS).reset_index(drop=True)
    common.to_csv(out / "odestream_common_keys.csv", index=False)
    ode = common.merge(ode, on=KEYS, validate="one_to_one")
    target_ref = common.merge(ref[KEYS + AC], on=KEYS, validate="one_to_one")
    label_error = float(np.abs(ode.actual_return_1.to_numpy() - target_ref.actual_return_1.to_numpy()).max())
    assert label_error < 1e-6
    ode["actual_return_1"] = target_ref.actual_return_1.to_numpy()
    ode.to_csv(point_dir / "ODEStream_backdated.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    secondary, secondary_daily, h1_rows = [], {}, []
    for name, full_frame in frames.items():
        f = common.merge(full_frame, on=KEYS, validate="one_to_one")
        r, d = evaluate(name, f, out, "ode_common_225")
        secondary.extend(r)
        secondary_daily.update({(name, mode): value for mode, value in d.items()})
        h1 = f[KEYS + [PC[0], AC[0]]].copy()
        h1["pred_signal"], h1["actual_signal"] = h1[PC[0]], h1[AC[0]]
        r, _ = evaluate(name, h1, out, "h1_ode_common_225", {"path": "h1"})
        h1_rows.extend(r)
    ode_eval = ode.copy()
    ode_eval["pred_signal"], ode_eval["actual_signal"] = ode.pred_return_1, ode.actual_return_1
    r, _ = evaluate("ODEStream_historical", ode_eval, out, "h1_ode_common_225", {"path": "h1"})
    h1_rows.extend(r)
    # A descriptive strategy table mixes signals but uses identical realized daily returns.
    secondary.extend([{**r[0], "signal": "h1_historical_timing_issue"}])
    pd.DataFrame(secondary).to_csv(out / "ode_common_225_strategy_metrics.csv", index=False)
    pd.DataFrame(h1_rows).to_csv(out / "h1_common_225_metrics.csv", index=False)
    save(out / "alignment_audit.json", {"status": "passed", "full_rows": 66103, "full_dates": 226,
         "full_models": audit, "odestream_common_rows": len(common), "odestream_common_dates": common.as_of_date.nunique(),
         "odestream_actual_return_1_max_error_before_canonicalization": label_error,
         "odestream_common_start": common.as_of_date.min(), "odestream_common_end": common.as_of_date.max(),
         "all_forecasts_finite": True, "common_target_source": source_entry(ref_path),
         "ode_note": "Backdated target date to previous observed stock trading day. ODE same-target-day sequential supervision prevents a causally matched comparison."})
    plot_curves(out, full_daily)
    fields = [("trajectory_rmse", "轨迹RMSE", ".6f"), ("direction_accuracy", "方向率", ".2%"),
              ("ic", "IC", ".6f"), ("rankic", "RankIC", ".6f"),
              ("net_cumulative_return", "净累计收益", ".3%"), ("sharpe", "Sharpe", ".3f"),
              ("max_drawdown", "最大回撤", ".3%")]
    h1table = pd.DataFrame(h1_rows)
    h1table["signal"] = "h1"
    report = ["# ODEStream、Chronos-2、TimesFM 3.0 与 HCA：预测恢复和统一重评",
              "", "日期：2026-11-05。仅使用已有预测，不训练、不改预测、不按本轮测试指标选模型。用户所说 chromos 对应本地 Chronos-2，tsfm 对应 TimesFM 3.0；TSFM本身也是时间序列基础模型的统称。",
              "", "## 找回的预测与归档",
              "", "Chronos-2、TimesFM 3.0各66,103行、226日，summary为complete；原始CSV各约182MB，含十步价格/收益及9分位数，已按原始字节打包在prediction_archives/。Kronos原始226个逐日文件67,390行，按固定键对齐到66,103行。ODEStream原始67,006行、226个目标日、297股，保留原文件。静态Prediction、普通在线A、HCA-PCA预测一并归档。ZIP通过逐成员CRC及SHA256核验，权重/原始行情不在这些预测包内。",
              "", "SSPT本地仅发现smoke结果，没有正式测试预测；不把smoke中的complete当作正式实验完成。DoubleAdapt此前正式结果不完整，继续标为无法验证。中断旧Chronos results-archive没有拼入最终输出。",
              "", "## 统一226日主表",
              "", "2025-07-01—2026-06-05，66,103个完全相同股票/信号日键。90→10累计收益目标；Chronos/TimesFM用原始q0.5价格路径，Kronos用原有采样聚合。所有模型统一十步标签（源差异先核验<1e-6）；Path十步均值，Endpoint第10步。Top30/Drop3/最少持有5日/单边15bps/252日，actual_return_1兑现次日收益，现金起始。原TSFM输出使用Top50及旧代理规则，本次重新用与HCA相同的evaluate计算，不能直接引用旧收益表替代。",
              "", "### Path", "", markdown_table(main_table, "path", fields),
              "", "### Endpoint", "", markdown_table(main_table, "endpoint", fields),
              "", "方向率对应当前信号的正负分类，轨迹RMSE为十步累计收益整体误差；价格误差与收益误差不混比。净累计收益不等于年化。",
              "", "![统一累计净收益](cumulative_returns.png)",
              "", "## ODEStream日期及在线时序问题",
              "", f"原始date是被预测目标日；由原始股票交易日回退到前一个收盘信号日，并核验current_close/true_close。与主面板公共范围为{common.as_of_date.min()}—{common.as_of_date.max()}，{common.as_of_date.nunique()}日、{len(common):,}键。原旧摘要66709是目标行回退日期、去掉7月1日目标后的行数，不是原CSV缺297行。",
              "", "更关键：运行器只有一套共享model/optimizer，按目标日→股票排序，每条predict_one后立即update_one读取该条目标标签。同目标日除第一股外，后续股票已接受此前股票同目标日真实收盘监督。若要求在前一收盘时统一发布截面预测，该标签尚未可用；所以保存的ODEStream不能作为与HCA时序公平的确认性基线。此轮没有重新训练修复它，也不能通过回退日期修复权重已受监督的问题。证据见odestream_audit.json、逐日更新顺序CSV和原运行器。",
              "", "以下仅历史诊断：同公共股票/信号日、相同H1收益目标和组合规则；ODEStream原始24→1，其他取90→10输出中的第一步。不得把ODEStream的H1误差标成十步轨迹误差。",
              "", markdown_table(h1table, "h1", [("rmse", "H1 RMSE", ".6f"), ("direction_accuracy", "H1方向率", ".2%"),
                              ("ic", "H1 IC", ".6f"), ("rankic", "H1 RankIC", ".6f"),
                              ("net_cumulative_return", "H1策略净收益", ".3%")]),
              "", "ode_common_225_strategy_metrics.csv另保留主模型Path/Endpoint与ODE H1策略的描述性同日收益表；信号目标仍不同，不是同任务预测精度排名。",
              "", "## 不确定性及论文使用边界",
              "", "paired_intervals_full_226.csv为配对日期块bootstrap：block10、5000次、seed20261013，重采样已实现日IC/收益，不重训练或重建持仓，未校正多重检验。收益较高不代表预测更准确或统计显著。",
              "", "Chronos/TimesFM为本项目零样本，HCA包含历史监督训练与在线反馈，训练预算/输入协变量不同；不能把差异全部归于HCA结构。预训练版本对历史测试时点的可用性和预训练数据重叠未排除。固定当期成分股、未来完整窗口可评价样本筛选、已反复查看的单市场单窗口等限制保留。",
              "", "新电脑git lfs pull后可直接解压预测包复算；normalized_points/保留点预测十步收益与统一标签，run_metadata/保留原配置、版本、完成状态和数据指纹。sources.json和archive_manifest.json提供来源及字节哈希。无需原始行情即可运行 python scripts/evaluation/recheck_archived_baseline_points.py --model Chronos2 --verify-all-zips。"]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save(out / "sources.json", sources)
    # Verify every compressed source against the original bytes.
    archives = []
    for zpath in sorted(archive_dir.glob("*.zip")):
        with zipfile.ZipFile(zpath) as archive:
            assert archive.testzip() is None
            members = []
            for entry in archive.infolist():
                with archive.open(entry) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                if zpath.stem == "Kronos_zero_shot":
                    original = MODELS / "zero_shot/predictions" / entry.filename
                elif zpath.stem == "ODEStream_original":
                    original = ode_path
                else:
                    original = paths[zpath.stem]
                assert h.hexdigest() == digest(original)
                members.append({"name": entry.filename, "bytes": entry.file_size, "sha256": h.hexdigest()})
        archives.append({"file": str(zpath.relative_to(out)).replace(chr(92), "/"),
                         "bytes": zpath.stat().st_size, "sha256": digest(zpath), "members": members})
    save(out / "archive_manifest.json", {"status": "verified", "archives": archives})
    print(main_table.to_string(index=False), flush=True)
    print("Recovered and verified " + str(len(archives)) + " forecast archives", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args())
