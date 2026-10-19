"""Strict, shared-panel evaluation of DoubleAdapt and locked historical streams."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate.inputs import normalize_predictions
from evaluate.metrics import EvaluationConfig, clean_json, evaluate_frame
from .data import sha256, write_json

ROOT=Path(__file__).resolve().parents[2]
KEYS=["as_of_date","stock"]
ACTUAL=[f"actual_return_{k}" for k in range(1,11)]


def align(frame, reference, *, mixed=False):
    frame=normalize_predictions(frame).sort_values(KEYS).reset_index(drop=True)
    reference=normalize_predictions(reference).sort_values(KEYS).reset_index(drop=True)
    if frame.duplicated(KEYS).any() or not frame[KEYS].equals(reference[KEYS]):
        raise AssertionError("Prediction keys differ; intersection comparisons are forbidden")
    for col in ACTUAL+["actual_signal","actual_endpoint_return"]:
        if col in frame and not np.allclose(frame[col],reference[col],rtol=0,atol=1e-7):
            raise AssertionError(f"Conflicting target column: {col}")
        frame[col]=reference[col].to_numpy()
    if mixed:
        frame["pred_signal"]=frame["final_signal"].to_numpy()
        frame["pred_endpoint_return"]=frame["pred_signal"]
    required=["pred_signal","actual_signal","pred_endpoint_return","actual_endpoint_return","actual_return_1"]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise FloatingPointError("Nonfinite evaluation input")
    return frame


def indices(n,repetitions=5000,block=10,seed=20261013):
    if n<block:raise ValueError("Too few dates for block bootstrap")
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-block+1,size=(repetitions,math.ceil(n/block)))
    return (starts[...,None]+np.arange(block)).reshape(repetitions,-1)[:,:n]


def return_stats(values):
    wealth=np.cumprod(1+values,axis=-1)
    peaks=np.maximum(1,np.maximum.accumulate(wealth,axis=-1))
    std=values.std(axis=-1,ddof=1)
    return {"net_cumulative_return":wealth[...,-1]-1,
            "sharpe":np.divide(values.mean(axis=-1),std,out=np.zeros_like(std),where=std>0)*np.sqrt(252),
            "max_drawdown":(wealth/peaks-1).min(axis=-1)}


def paired_statistics(left,right,draws,label):
    if not left.as_of_date.equals(right.as_of_date):raise AssertionError("Bootstrap date mismatch")
    rows=[]
    for metric in ["ic","rankic","endpoint_ic","endpoint_rankic","top_net"]:
        difference=left[metric].to_numpy()-right[metric].to_numpy()
        boot=difference[draws].mean(axis=1)
        rows.append({"comparison":label,"metric":"mean_"+metric,"delta":difference.mean(),
                     "ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975)})
    la,ra=left.top_net.to_numpy(),right.top_net.to_numpy()
    lp,rp=return_stats(la),return_stats(ra)
    lb,rb=return_stats(la[draws]),return_stats(ra[draws])
    for metric in lp:
        boot=lb[metric]-rb[metric]
        rows.append({"comparison":label,"metric":metric,"delta":lp[metric]-rp[metric],
                     "ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975)})
    return rows


def compare(predictions,manifest_path,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError("Evaluation output must be new")
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frames,sources={},{}
    for name,entry in manifest.items():
        path=Path(entry["path"])
        if not path.exists():
            method={"Page1_C_Static":"prediction_adapter","D_RankIC_Static":"adapter_drankic_dynamic_static_20261013",
                    "D_RankIC_Continual":"adapter_drankic_dynamic_branch_20261013",
                    "D_RankIC_Dynamic":"adapter_drankic_dynamic_continual_20261013"}[name]
            path=ROOT/"experiments/csi300_aligned/models"/method/"predictions.csv"
        if sha256(path)!=entry["sha256"]:raise AssertionError("Locked baseline hash mismatch")
        sources[name]={"path":str(path.resolve()),"sha256":entry["sha256"]}
        frames[name]=pd.read_csv(path,dtype={"stock":str})
    static_cfg=json.loads((Path(sources["D_RankIC_Static"]["path"]).parent/"run_config.json").read_text(encoding="utf-8"))
    continual_cfg=json.loads((Path(sources["D_RankIC_Continual"]["path"]).parent/"run_config.json").read_text(encoding="utf-8"))
    if static_cfg["hashes"]["initial_checkpoint"]!=continual_cfg["expected_initial_checkpoint_sha256"] or continual_cfg["hard_acceptance_gate"]:
        raise AssertionError("D reference is not the locked paired no-gate experiment")
    for source in predictions:
        source=Path(source)
        cfg=json.loads((source.parent/"run_config.json").read_text(encoding="utf-8"))
        status=json.loads((source.parent/"status.json").read_text(encoding="utf-8"))
        if cfg["smoke"] or status["status"] not in ("predictions_complete","complete"):
            raise AssertionError("Incomplete/smoke experiment cannot enter formal comparison")
        name=f"DoubleAdapt_seed{cfg['seed']}"
        if name in frames:raise ValueError("Duplicate seed")
        frames[name]=pd.read_csv(source,dtype={"stock":str})
        sources[name]={"path":str(source.resolve()),"sha256":sha256(source)}
    reference=normalize_predictions(frames["Page1_C_Static"]).sort_values(KEYS).reset_index(drop=True)
    if len(reference)!=66103 or reference.as_of_date.nunique()!=226:
        raise AssertionError("Reference is not the fixed panel")
    frames={name:align(frame,reference,mixed=name=="D_RankIC_Dynamic") for name,frame in frames.items()}
    output.mkdir(parents=True)
    write_json(output/"sources.json",sources)
    config=EvaluationConfig(dataset="CSI300",top_k=30,drop_n=3,cost_rate=.0015,
                            min_holding_days=5,bootstrap_reps=5000,bootstrap_block=10,seed=20261013)
    daily,metrics={},[]
    for name,frame in frames.items():
        result=evaluate_frame(frame,config)
        if name=="D_RankIC_Dynamic":
            unavailable={"status":"不适用","reason":"dimensionless rank mixture; no return-valued prediction"}
            for mode in ("path","endpoint"):
                result["metrics"][mode]["point"]=unavailable
                result["metrics"][mode]["direction"]=unavailable
            result["endpoint_interpretation"]="final mixed score ranked against actual day-10 return; no 10-step forecast"
        directory=output/name;directory.mkdir()
        combined=pd.DataFrame(result["daily"]["path"]["cross_section"])
        portfolio=pd.DataFrame(result["daily"]["path"]["portfolio"])
        endpoint=pd.DataFrame(result["daily"]["endpoint"]["cross_section"]).rename(columns={"ic":"endpoint_ic","rankic":"endpoint_rankic"})
        combined=combined.merge(portfolio[["as_of_date","top_net","turnover","transaction_cost"]],on="as_of_date",validate="one_to_one")
        combined=combined.merge(endpoint[["as_of_date","endpoint_ic","endpoint_rankic"]],on="as_of_date",validate="one_to_one").sort_values("as_of_date").reset_index(drop=True)
        if len(combined)!=226 or not np.isfinite(combined[["ic","rankic","top_net","endpoint_rankic"]]).all().all():
            raise AssertionError("Incomplete/nonfinite daily evaluation")
        daily[name]=combined
        combined.to_csv(directory/"daily.csv",index=False)
        portfolio.to_csv(directory/"portfolio_daily.csv",index=False)
        monthly=combined.assign(month=combined.as_of_date.str[:7]).groupby("month").agg(
            rankic=("rankic","mean"),ic=("ic","mean"),endpoint_rankic=("endpoint_rankic","mean"),
            net_return=("top_net",lambda v:(1+v).prod()-1),turnover=("turnover","mean"))
        monthly.to_csv(directory/"monthly.csv")
        write_json(directory/"evaluation_summary.json",clean_json(result))
        row={"method":name,**return_stats(combined.top_net.to_numpy()),"path_ic":combined.ic.mean(),
             "path_rankic":combined.rankic.mean(),"endpoint_ic":combined.endpoint_ic.mean(),
             "endpoint_rankic":combined.endpoint_rankic.mean(),"turnover":combined.turnover.mean(),
             "cost_sum":combined.transaction_cost.sum()}
        if name!="D_RankIC_Dynamic":
            error=frame[[f"pred_return_{k}" for k in range(1,11)]].to_numpy()-frame[ACTUAL].to_numpy()
            row.update(trajectory_mae=np.abs(error).mean(),trajectory_rmse=np.sqrt(np.square(error).mean()))
        metrics.append(row)
        print(json.dumps({"stage":"evaluated","method":name,"rankic":row["path_rankic"]}),flush=True)
    pairs=indices(226)
    paired=[]
    for name in daily:
        if name.startswith("DoubleAdapt"):
            for baseline in manifest:
                paired+=paired_statistics(daily[name],daily[baseline],pairs,f"{name}_minus_{baseline}")
    pd.DataFrame(metrics).to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(paired).to_csv(output/"paired_bootstrap.csv",index=False)
    seeds=pd.DataFrame(metrics);seeds=seeds[seeds.method.str.startswith("DoubleAdapt")]
    seeds.drop(columns="method").agg(["mean","std"]).to_csv(output/"seed_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for name,frame in daily.items():
        dates=pd.to_datetime(frame.as_of_date)
        axes[0].plot(dates,(np.cumprod(1+frame.top_net)-1)*100,label=name)
        axes[1].plot(dates,frame.rankic.rolling(20,min_periods=1).mean(),label=name)
    axes[0].set_ylabel("Net cumulative return (%)");axes[1].set_ylabel("Path RankIC (20-day mean)")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"comparison.png",dpi=170);fig.savefig(output/"comparison.svg");plt.close(fig)
    lines=["# DoubleAdapt CSI300 90→10 比较", "", "固定协议的回顾性在线比较；测试期此前已查看。",
           "", "所有方法共享226日、66,103键、真实标签、Top30/Drop3、最低持有5日及15bps成本。",
           "DoubleAdapt使用独立GRU、五字段及真实收益路径，并在IC选模后更新验证期；Kronos有不同预训练资源、损失和更新数据。差异评价整套方法，不单独归因于持续学习。",
           "", "| 方法 | 净累计收益 | Sharpe | 最大回撤 | Path RankIC | Endpoint RankIC |", "|---|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['method']} | {row['net_cumulative_return']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['path_rankic']:.5f} | {row['endpoint_rankic']:.5f} |")
    lines += ["", "配对95%区间见 paired_bootstrap.csv：共享非循环日期块、block=10、5,000次、seed=20261013；收益类统计重采样已实现收益，不重建持仓。多个对照的区间为探索性，未校正多重检验。",
              "多个DoubleAdapt种子与同一历史Kronos seed=100比较，不声称Kronos多种子稳健性。维度无量纲的动态混合不计算点预测或方向指标。",
              "", "![比较曲线](comparison.png)"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"status.json",{"status":"complete","methods":list(frames),"rows":66103,"dates":226,
        "bootstrap":{"block":10,"repetitions":5000,"seed":20261013},"retrospective":True})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,nargs="+",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    compare(args.predictions,args.manifest,args.output_dir)


if __name__=="__main__":main()
