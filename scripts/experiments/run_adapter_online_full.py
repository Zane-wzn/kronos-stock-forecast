"""Full locked 226-day A adaptation from historical Prediction/RankIC adapters."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.experiments import run_online_module_pilot as pilot
from src.doubleadapt.compare import indices, return_stats

KEYS, PCOLS, ACOLS = pilot.KEYS, pilot.PCOLS, pilot.ACOLS
MODELS = {
    "Prediction": {
        "checkpoint": ROOT / "experiments/csi300_aligned/models/prediction_adapter/model.pt",
        "checkpoint_sha256": "111f72975e77e165211b441a731b4da17dca4e26629b848839ce13aaffcd94eb",
        "reference": ROOT / "experiments/csi300_aligned/models/prediction_adapter/predictions.csv",
    },
    "RankIC": {
        "checkpoint": ROOT / ".runtime/D_model_rankic_pilot_original.pt",
        "checkpoint_sha256": pilot.old.EXPECTED_CKPT,
        "reference": ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_static_20261013/predictions.csv",
    },
}
HISTORY = {
    "Main_Continual": ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_branch_20261013/predictions.csv",
    "Main_Hedge": ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_continual_20261013/predictions.csv",
}


def load_data():
    cache = ROOT / "experiments/csi300_aligned/doubleadapt_cache_v2"
    reference = ROOT / "experiments/csi300_aligned/reference"
    meta = pd.read_csv(cache / "meta.csv", dtype={"stock": str})
    test = meta[meta.split.eq("test")].reset_index(drop=True)
    hidden = np.load(reference / "hidden_cache/hidden_test.npy", mmap_mode="r")
    values = np.load(cache / "y.npy", mmap_mode="r")
    assert hidden.shape == (66103, 512) and values.shape == (len(meta), 10)
    np.testing.assert_array_equal(meta.row_id.to_numpy(), np.arange(len(meta)))
    dates = sorted(test.as_of_date.unique())
    assert len(dates) == 226 and dates[0] == "2025-07-01" and dates[-1] == "2026-06-05"
    assert not test.duplicated(KEYS).any()
    hidden_hash = pilot.old.sha256(reference / "hidden_cache/hidden_test.npy")
    assert hidden_hash == "20e91f77ac595f119885a99f7a3c7f96672403879e9a85213bd5b9d55f349310"
    windows = pd.read_csv(reference / "date_windows.csv", dtype=str).drop_duplicates("as_of_date").set_index("as_of_date")
    label_dates = {d: str(windows.loc[d, "label_dates"]).split("|") for d in dates}
    stocks = sorted(set(meta.loc[meta.split.eq("train"), "stock"]), key=pilot.old.stable_stock_key)[:60]
    rows, h_by_date, training = {}, {}, {}
    for d in dates:
        part = test[test.as_of_date.eq(d)].copy().sort_values("stock")
        ids = part.index.to_numpy()
        part = part.reset_index(drop=True)
        part["label_dates"] = [label_dates[d]] * len(part)
        assert len(label_dates[d]) == 10 and label_dates[d][0] > d
        assert part.label_start_date.eq(label_dates[d][0]).all() and part.label_end_date.eq(label_dates[d][-1]).all()
        rows[d] = part
        h_by_date[d] = torch.tensor(np.asarray(hidden[ids]), dtype=torch.float32)
        training[d] = part[part.stock.isin(stocks)].reset_index(drop=True)
    schedule = pilot.make_schedule(dates, training)
    releases = {int(r.row_id): r.label_dates for f in rows.values() for r in f.itertuples()}
    return meta, test, hidden, values, dates, stocks, rows, h_by_date, schedule, releases, hidden_hash


def attach_targets(prediction, reference, name):
    frame = prediction.merge(reference[KEYS + ACOLS], on=KEYS, validate="one_to_one")
    assert len(frame) == 66103
    frame = frame.drop(columns=["group"], errors="ignore")
    frame["method"] = name
    frame["seed"] = 100
    for prefix, cols in (("pred", PCOLS), ("actual", ACOLS)):
        frame[f"{prefix}_signal"] = frame[cols].mean(axis=1)
        frame[f"{prefix}_path_mean_return"] = frame[f"{prefix}_signal"]
        frame[f"{prefix}_endpoint_return"] = frame[cols[-1]]
    return frame


def audit_stream(out, model_name, group, result, initial, final, dates, reference):
    prediction, access, updates, _ = result
    ordered = prediction.sort_values(KEYS).reset_index(drop=True)
    assert ordered[KEYS].equals(reference[KEYS]) and np.isfinite(ordered[PCOLS].to_numpy()).all()
    if group == "S":
        error = float(np.abs(ordered[PCOLS].to_numpy() - reference[PCOLS].to_numpy()).max())
        assert error <= 1e-7
    else:
        assert len(updates) == 216 and len(access) == 69120
        assert updates.date.tolist() == dates[10:]
        assert (access.complete_date <= access.update_date).all() and (access.label_date <= access.update_date).all()
        assert (access.origin_date < access.update_date).all() and not access.duplicated(["row_id", "horizon"]).any()
        assert updates.supervision.eq(320).all() and updates.optimizer_steps.eq(1).all()
        assert (updates.adapter_grad_norm > 0).all()
        for d, digest in access[["update_date", "prediction_sha256"]].drop_duplicates().itertuples(index=False, name=None):
            assert pilot.old.sha256(out / model_name / "prediction_shards" / group / f"{d}.csv") == digest
        first_versions = prediction.groupby("as_of_date").pred_version.agg(["min", "max"])
        expected = np.maximum(0, np.arange(226) - 10)
        np.testing.assert_array_equal(first_versions["min"], expected)
        np.testing.assert_array_equal(first_versions["max"], expected)
        error = None
    delta = {part: math.sqrt(sum(float(((a.detach() - b.detach()) ** 2).sum())
        for a, b in zip(getattr(initial, part).parameters(), getattr(final, part).parameters()))) for part in ("adapter", "head")}
    if group == "S":
        assert delta == {"adapter": 0, "head": 0}
    else:
        assert delta["adapter"] > 0 and delta["head"] > 0
    return {"model": model_name, "group": group, "rows": len(prediction), "updates": len(updates),
        "supervision": len(access), "static_max_abs_error": error, "parameter_delta_l2": delta,
        "future_reads": 0, "duplicate_consumption": 0, "status": "passed"}


def evaluate(out, names, dates):
    for window, start in (("full_226", dates[0]), ("post_warmup_216", dates[10])):
        command = [sys.executable, "-m", "evaluate", "--predictions",
            *[str(out / "streams" / name / "predictions.csv") for name in names],
            "--output-dir", str(out / "evaluation" / window), "--dataset", "CSI300",
            "--test-start", start, "--test-end", dates[-1], "--reference-keys", str(out / "reference_keys.csv"),
            "--top-k", "30", "--drop-n", "3", "--min-holding-days", "5", "--cost-rate", ".0015",
            "--annualization-days", "252", "--bootstrap-reps", "5000", "--bootstrap-block", "10", "--seed", "20261013"]
        subprocess.run(command, cwd=ROOT, check=True)


def report(out, names, dates):
    daily, summaries, monthly = {}, [], []
    for window, selected_dates in (("full_226", dates), ("post_warmup_216", dates[10:])):
        for name in names:
            directory = out / "evaluation" / window / f"{name}_predictions"
            summary = json.loads((directory / "evaluation_summary.json").read_text(encoding="utf-8"))
            for signal in ("path", "endpoint"):
                cross = pd.read_csv(directory / f"evaluation_{signal}_cross_section_daily.csv")
                portfolio = pd.read_csv(directory / f"evaluation_{signal}_portfolio_daily.csv")
                if cross.empty:
                    continue  # Hedge has no return-valued endpoint prediction.
                frame = cross.merge(portfolio, on="as_of_date", validate="one_to_one", suffixes=("", "_portfolio")).sort_values("as_of_date").reset_index(drop=True)
                assert frame.as_of_date.tolist() == selected_dates
                assert np.isfinite(frame[["ic", "rankic", "top_net"]].to_numpy()).all()
                np.testing.assert_allclose(frame.top_net, frame.top_raw - .0015 * frame.turnover, rtol=0, atol=1e-14)
                daily[window, name, signal] = frame
                row = {"window": window, "model": name, "signal": signal, "dates": len(frame),
                    "ic": float(frame.ic.mean()), "rankic": float(frame.rankic.mean()), "turnover": float(frame.turnover.mean()),
                    "annualized_excess_proxy": float((frame.top_net-frame.benchmark_daily_return).mean()*252),
                    "total_transaction_cost_rate": float(frame.transaction_cost.sum()),
                    **{k: float(v) for k, v in return_stats(frame.top_net.to_numpy()).items()}}
                row["trajectory_mae"] = summary["metrics"]["trajectory_pooled"].get("mae")
                row["trajectory_rmse"] = summary["metrics"]["trajectory_pooled"].get("rmse")
                summaries.append(row)
                mm = frame.assign(month=frame.as_of_date.str[:7]).groupby("month").agg(ic=("ic", "mean"), rankic=("rankic", "mean"),
                    net_cumulative_return=("top_net", lambda x: (1+x).prod()-1), turnover=("turnover", "mean")).reset_index()
                mm["window"], mm["model"], mm["signal"] = window, name, signal
                monthly.append(mm)
    summary = pd.DataFrame(summaries)
    pilot.old.atomic_csv(out / "comparison_metrics.csv", summary)
    pilot.old.atomic_csv(out / "monthly_metrics.csv", pd.concat(monthly, ignore_index=True))
    comparisons = [("Prediction_A", "Prediction_Static"), ("RankIC_A", "RankIC_Static"),
                   ("Prediction_A", "RankIC_A"), ("RankIC_A", "Main_Continual"),
                   ("RankIC_A", "Main_Hedge"), ("Prediction_A", "Main_Hedge")]
    boot_rows, pairs = [], []
    for window, selected_dates in (("full_226", dates), ("post_warmup_216", dates[10:])):
        draws = indices(len(selected_dates), 5000, 10, 20261013)
        for a, b in comparisons:
            for signal in ("path", "endpoint"):
                if (window, a, signal) not in daily or (window, b, signal) not in daily:
                    continue
                aa, bb = daily[window, a, signal], daily[window, b, signal]
                diff = pd.DataFrame({"window": window, "comparison": a + "_minus_" + b, "signal": signal, "as_of_date": aa.as_of_date})
                for metric in ("ic", "rankic", "top_net", "turnover", "top_raw", "transaction_cost"):
                    delta = aa[metric].to_numpy() - bb[metric].to_numpy()
                    diff[metric + "_delta"] = delta
                    boot = delta[draws].mean(1)
                    boot_rows.append({"window": window, "comparison": a + "_minus_" + b, "signal": signal, "metric": "mean_" + metric,
                        "delta": float(delta.mean()), "ci_low": float(np.quantile(boot,.025)), "ci_high": float(np.quantile(boot,.975))})
                pairs.append(diff)
                for metric, value in return_stats(aa.top_net.to_numpy()).items():
                    base = return_stats(bb.top_net.to_numpy())[metric]
                    boot = return_stats(aa.top_net.to_numpy()[draws])[metric] - return_stats(bb.top_net.to_numpy()[draws])[metric]
                    boot_rows.append({"window": window, "comparison": a + "_minus_" + b, "signal": signal, "metric": metric,
                        "delta": float(value-base), "ci_low": float(np.quantile(boot,.025)), "ci_high": float(np.quantile(boot,.975))})
    pilot.old.atomic_csv(out / "paired_daily_differences.csv", pd.concat(pairs, ignore_index=True))
    pilot.old.atomic_csv(out / "paired_block10_bootstrap.csv", pd.DataFrame(boot_rows))
    draw_plots(out, names, daily)
    primary = summary[summary.window.eq("full_226")].set_index(["model", "signal"])
    lines = ["# A式在线适配：Prediction/RankIC完整226日比较", "",
        "主窗口：2025-07-01—2026-06-05，226日、66,103个股票日期键；所有组合从同一天现金建仓并计入首日成本。",
        "前10日未收到完整10步标签所以不更新，但计入主表。216日另从现金独立回测，仅作为辅助，不与旧226日结果混比。", "",
        "A沿用小实验协议：冻结Kronos，每日原Adapter+预测头联合更新一次，head_lr=1e-4、adapter_lr=1e-5，SmoothL1(beta=1)、AdamW无weight decay、clip=1、Dropout关闭。",
        "保留原训练期可用性+hash预选60股（覆盖59股），各horizon最多32个完整成熟样本；仅延长时间，不扩大训练队列。无replay、rank loss、gate、Hedge或测试期调参。",
        "", "|模型|信号|累计净收益|IC|RankIC|Sharpe|最大回撤|平均换手|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (name, signal), row in primary.iterrows():
        lines.append(f"|{name}|{signal}|{row.net_cumulative_return:.3%}|{row.ic:.6f}|{row.rankic:.6f}|{row.sharpe:.3f}|{row.max_drawdown:.3%}|{row.turnover:.3%}|")
    lines += ["", "同checkpoint的在线−静态为主要归因比较；Prediction_A与RankIC_A是不同离线起点，不能把两者差异归因于在线机制。Main_Continual/Main_Hedge为历史文件重评，不重新训练。Hedge只有无量纲Path混合分数，未伪造Endpoint和收益单位点预测指标。", "",
        "|比较|信号|累计净收益差（百分点）|IC差|RankIC差|", "|---|---|---:|---:|---:|"]
    for a,b in comparisons:
        for signal in ("path", "endpoint"):
            if (a,signal) in primary.index and (b,signal) in primary.index:
                aa,bb=primary.loc[(a,signal)],primary.loc[(b,signal)]
                lines.append(f"|{a}−{b}|{signal}|{100*(aa.net_cumulative_return-bb.net_cumulative_return):+.3f}|{aa.ic-bb.ic:+.6f}|{aa.rankic-bb.rankic:+.6f}|")
    lines += ["", "|在线−自身静态|信号|指标|差值|配对95%下限|配对95%上限|", "|---|---|---|---:|---:|---:|"]
    for row in boot_rows:
        if row["window"] == "full_226" and row["comparison"] in ("Prediction_A_minus_Prediction_Static", "RankIC_A_minus_RankIC_Static") and row["metric"] in ("mean_ic", "net_cumulative_return"):
            scale=100 if row["metric"] == "net_cumulative_return" else 1
            label="净收益差（百分点）" if scale==100 else "IC差"
            lines.append(f"|{row['comparison']}|{row['signal']}|{label}|{row['delta']*scale:+.6f}|{row['ci_low']*scale:+.6f}|{row['ci_high']*scale:+.6f}|")
    lines += ["", "评测调用原python -m evaluate：Top30/Drop3，最低持有5日，单向15bps，actual_return_1兑现日收益，年化252。IC=逐日Pearson，RankIC=逐日Spearman。",
        "区块bootstrap：block=10、5000次、seed=20261013，共享日期块。重采样已实现日收益，不重模拟持仓/训练。未做多重比较校正，区间为探索性证据。",
        "全期为已多次查看的旧固定测试面板，保留成分股前视/幸存者偏差。A由其中60日小实验筛选，扩展全期不构成未触碰独立确认。", "",
        "![完整期累计净收益](cumulative_net_returns.png)", "", "![完整期IC](ic_curves.png)", "",
        "comparison_metrics.csv包含全部226/216日指标，monthly_metrics.csv包含逐月结果；paired_block10_bootstrap.csv、paired_daily_differences.csv包含配对证据。audit.json、input_manifest.json和source快照记录运行身份。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def draw_plots(out, names, daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind, filename in (("returns", "cumulative_net_returns"), ("ic", "ic_curves")):
        fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
        for ax,signal in zip(axes,("path","endpoint")):
            for name in names:
                if ("full_226",name,signal) not in daily:
                    continue
                frame=daily["full_226",name,signal]
                values=(np.cumprod(1+frame.top_net)-1)*100 if kind=="returns" else frame.ic.rolling(20,min_periods=1).mean()
                ax.plot(pd.to_datetime(frame.as_of_date),values,label=name,lw=1.5)
            ax.axhline(0,color="gray",lw=.7)
            ax.set_ylabel(signal+": "+("net return (%)" if kind=="returns" else "IC (20-day mean)"))
            ax.legend(ncol=2,fontsize=8); ax.grid(alpha=.2)
        fig.suptitle("CSI300 | 226 days | same locked panel | retrospective")
        fig.tight_layout()
        for ext in ("png","svg"):
            fig.savefig(out/f"{filename}.{ext}",dpi=170)
        plt.close(fig)


def main(args):
    started=time.perf_counter()
    checks=pilot.correctness_checks()
    print(json.dumps(checks),flush=True)
    pilot.old.seed_all(100)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("fresh output directory required")
    meta,test,hidden,values,dates,stocks,rows,h_by_date,schedule,releases,hidden_hash=load_data()
    out.mkdir(parents=True,exist_ok=True)
    pilot.old.atomic_json(out/"correctness_report.json",checks)
    sources={name:{"path":str(path),"sha256":pilot.old.sha256(path)} for name,path in {
        **{name+"_checkpoint":entry["checkpoint"] for name,entry in MODELS.items()},
        **{name+"_reference":entry["reference"] for name,entry in MODELS.items()},**HISTORY,
        "runner":Path(__file__),"engine":Path(pilot.__file__),"original_engine":Path(pilot.old.__file__),
        "meta":ROOT/"experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv",
        "labels":ROOT/"experiments/csi300_aligned/doubleadapt_cache_v2/y.npy"}.items()}
    pilot.old.atomic_json(out/"input_manifest.json",sources)
    pilot.old.atomic_json(out/"run_config.json",{"dates":dates,"rows":66103,"seed":100,"hidden_sha256":hidden_hash,
        "training_stocks":stocks,"training_pool_policy":"unchanged pilot train availability/hash cohort; F/P intersection retained",
        "training_rows_per_horizon":32,"head_lr":1e-4,"adapter_lr":1e-5,"dropout":"disabled",
        "optimizer":"AdamW","weight_decay":0,"gradient_clip":1,"loss":"SmoothL1 beta1","steps_per_day":1,
        "primary_window":"all226 dates; cash start 2025-07-01","warmup_days":10,"updates":216,
        "secondary_window":"post_warmup216; independent cash start","retrospective":True,"no_tuning":True})
    for name,path in (("executed_source.py",__file__),("executed_engine.py",pilot.__file__),("executed_original_engine.py",pilot.old.__file__)):
        shutil.copy2(path,out/name)
    pilot.old.atomic_csv(out/"training_schedule.csv",pd.DataFrame([{"update_date":d,**e} for d,ee in schedule.items() for e in ee]))
    pilot.old.atomic_csv(out/"reference_keys.csv",test[KEYS].sort_values(KEYS))
    pilot.old.atomic_json(out/"status.json",{"status":"running"})
    audits,stats,names,online_records=[],[],[],{}
    reference_targets=None
    for model_name,entry in MODELS.items():
        assert pilot.old.sha256(entry["checkpoint"])==entry["checkpoint_sha256"]
        reference=pd.read_csv(entry["reference"],dtype={"stock":str}).sort_values(KEYS).reset_index(drop=True)
        assert reference[KEYS].equals(test[KEYS].sort_values(KEYS).reset_index(drop=True))
        lookup=test[KEYS+["row_id"]].merge(reference[KEYS+ACOLS],on=KEYS,validate="one_to_one")
        label_error=float(np.abs(np.asarray(values[lookup.row_id.to_numpy()])-lookup[ACOLS].to_numpy()).max())
        assert label_error<=1e-7
        if reference_targets is None:
            reference_targets=reference[KEYS+ACOLS]
        else:
            np.testing.assert_allclose(reference_targets[ACOLS],reference[ACOLS],atol=1e-7,rtol=0)
        initial=pilot.old.make_model(entry["checkpoint"],False)
        with torch.no_grad():
            z_by_date={d:initial.adapter(h_by_date[d]) for d in dates}
        # Block any online run until static identity has been checked across all dates.
        with torch.no_grad():
            pre=pd.concat([pd.concat([rows[d][KEYS].reset_index(drop=True),pd.DataFrame(initial.head(z_by_date[d]).numpy(),columns=PCOLS)],axis=1) for d in dates]).sort_values(KEYS)
        static_error=float(np.abs(pre[PCOLS].to_numpy()-reference[PCOLS].to_numpy()).max())
        assert static_error<=1e-7
        print(json.dumps({"stage":"preflight","model":model_name,"static_error":static_error,"label_error":label_error}),flush=True)
        for group,suffix in (("S","Static"),("A","A")):
            name=model_name+"_"+suffix; names.append(name)
            model=copy.deepcopy(initial); t=time.perf_counter()
            result=pilot.run_stream(group,model,h_by_date,z_by_date,rows,schedule,pilot.ReleasedLabels(values,releases),out/model_name)
            audit=audit_stream(out,model_name,group,result,initial,model,dates,reference)
            audits.append(audit)
            stats.append({"model":name,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
                "seconds":time.perf_counter()-t,"updates":audit["updates"],"supervision":audit["supervision"],
                "adapter_delta_l2":audit["parameter_delta_l2"]["adapter"],"head_delta_l2":audit["parameter_delta_l2"]["head"]})
            if group=="A":
                online_records[model_name]=result[1]
                if model_name=="RankIC":
                    previous=pd.read_csv(ROOT/"experiments/csi300_aligned/models/online_module_pilot_20261020/A/predictions.csv")
                    overlap=previous.merge(result[0],on=KEYS,validate="one_to_one",suffixes=("_old","_new"))
                    assert len(overlap)==len(previous)
                    max_error=max(float(np.abs(overlap[c+"_old"]-overlap[c+"_new"]).max()) for c in PCOLS)
                    assert max_error<=1e-7
                    audit["pilot_first60_max_abs_error"]=max_error
            frame=attach_targets(result[0],reference,name)
            pilot.old.atomic_csv(out/"streams"/name/"predictions.csv",frame)
            print(json.dumps({"stage":"stream_complete",**stats[-1]}),flush=True)
    pd.testing.assert_frame_equal(online_records["Prediction"].drop(columns="prediction_sha256"),online_records["RankIC"].drop(columns="prediction_sha256"))
    for name,path in HISTORY.items():
        history=pd.read_csv(path,dtype={"stock":str}).sort_values(KEYS).reset_index(drop=True)
        assert history[KEYS].equals(reference_targets[KEYS])
        if name=="Main_Hedge":
            frame=history[KEYS+["final_signal"]].rename(columns={"final_signal":"pred_signal"}).merge(reference_targets,on=KEYS,validate="one_to_one")
            frame["actual_signal"]=frame[ACOLS].mean(1)
            frame["method"]=name
            # No pred_return_10: the normalizer must leave endpoint unavailable.
        else:
            np.testing.assert_allclose(history[ACOLS],reference_targets[ACOLS],atol=1e-7,rtol=0)
            frame=history.copy(); frame["method"]=name
        pilot.old.atomic_csv(out/"streams"/name/"predictions.csv",frame)
        names.append(name)
    pilot.old.atomic_csv(out/"training_summary.csv",pd.DataFrame(stats))
    pilot.old.atomic_json(out/"audit.json",{"status":"passed","streams":audits,"same_training_events_across_checkpoints":True})
    evaluate(out,names,dates)
    report(out,names,dates)
    # Inputs must remain byte-identical throughout the run.
    for source in sources.values():
        assert pilot.old.sha256(Path(source["path"]))==source["sha256"]
    pilot.old.atomic_json(out/"status.json",{"status":"complete","seconds":time.perf_counter()-started,"dates":226,"rows_per_stream":66103,"streams":names})
    files=[p for p in out.rglob("*") if p.is_file() and "checkpoints" not in p.parts]
    pilot.old.atomic_json(out/"artifact_manifest.json",{str(p.relative_to(out)):{"bytes":p.stat().st_size,"sha256":pilot.old.sha256(p)} for p in files})
    print(f"COMPLETE {out}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"experiments/csi300_aligned/models/adapter_online_full_20261020")
    args=parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        # Never overwrite a completed experiment's status on an accidental rerun.
        status=args.output_dir/"status.json"
        if status.exists() and json.loads(status.read_text()).get("status")=="running":
            pilot.old.atomic_json(status,{"status":"failed","error":repr(exc)})
        raise
