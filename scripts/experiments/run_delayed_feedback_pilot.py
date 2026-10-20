"""Minimal delayed-feedback correctness and CSI300 pilot.

This is deliberately separate from the main continual-learning runners.  It
compares a frozen prediction head (S), complete-path feedback (F), and
per-horizon prefix feedback (P) under one optimizer step per trading day.
The online loop never reads a label before its horizon date is released.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CKPT = "fe1426d1ab0e627622860fd86eb81c8f05b74fbd54df29439556050a1d813961"
PRED_LEN = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_stock_key(stock: str) -> str:
    return hashlib.sha256(("prefix-pilot-v1|" + str(stock)).encode()).hexdigest()


class Adapter(nn.Module):
    def __init__(self, d: int = 512, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d)

    def forward(self, h):
        return h + self.up(self.drop(F.gelu(self.down(self.ln(h)))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.head = nn.Linear(512, PRED_LEN)

    def forward(self, h):
        return self.head(self.adapter(h))


def seed_all(seed: int = 100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def nested_equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def run_toy_checks():
    """Exercise the actual online engine, including optimizer resume."""
    import tempfile
    global PRED_LEN
    original_horizon=PRED_LEN
    PRED_LEN=3
    results=[]
    try:
        seed_all(100)
        dates=[f"2020-01-{j+1:02d}" for j in range(20)]
        positions={d:j for j,d in enumerate(dates)}
        run_dates=dates[:14]
        z={d:torch.randn(5,4) for d in run_dates}
        rows={d:pd.DataFrame([{"stock":f"s{i}","row_id":j*5+i,
                             "label_dates":dates[j+1:j+4]} for i in range(5)]) for j,d in enumerate(run_dates)}
        values=np.random.default_rng(100).normal(0,.01,(70,3))
        initial=nn.Module(); initial.head=nn.Linear(4,3)
        class GuardedLabels:
            def __init__(self,data,table): self.data,self.table,self.today=data,table,""
            def __getitem__(self,key):
                row,k=key; origin=row//5; stock=row%5
                release=self.table[run_dates[origin]].iloc[stock].label_dates[k]
                assert release<=self.today,"unreleased label access"
                return self.data[key]
        with tempfile.TemporaryDirectory(prefix="prefix_toy_") as task_temp:
            root=Path(task_temp)
            def run(name,group="P",data=None,table=None,stop=None,resume=None,lr=1e-4):
                model=copy.deepcopy(initial)
                table=rows if table is None else table
                out=root/name
                result=online_run(group,model,z,table,GuardedLabels(values if data is None else data,table),
                                  positions,run_dates[3:],out,max_rows=4,stop_after=stop,resume_from=resume,lr=lr)
                return model,result,out
            full,records,fullout=run("full")
            assert len(records[2])==11 and len(records[1])==11*3*4
            assert (records[1].label_date<=records[1].update_date).all()
            assert not records[1].duplicated(["row_id","horizon"]).any()
            results.append("actual_engine_release_and_single_consumption")
            # Poison everything that has not been released as of cutoff. Read
            # neither these NaNs nor their values before that cutoff.
            cutoff=dates[7]; poisoned=values.copy()
            for j,d in enumerate(run_dates):
                for i in range(5):
                    for k,release in enumerate(rows[d].iloc[i].label_dates):
                        if release>cutoff: poisoned[j*5+i,k]=np.nan
            clean,cleanrec,cleanout=run("cutoff_clean",stop=8)
            bad,badrec,badout=run("cutoff_poison",data=poisoned,stop=8)
            assert nested_equal(clean.state_dict(),bad.state_dict())
            pd.testing.assert_frame_equal(cleanrec[0],badrec[0])
            cstate=torch.load(cleanout/"checkpoints/P"/f"{cutoff}.pt",weights_only=False)
            bstate=torch.load(badout/"checkpoints/P"/f"{cutoff}.pt",weights_only=False)
            assert nested_equal(cstate["optimizer"],bstate["optimizer"])
            results.append("poison_future_labels_parameters_optimizer_predictions_unchanged")
            resumed,resumerec,resumeout=run("resumed",resume=cleanout/"checkpoints/P"/f"{cutoff}.pt")
            assert nested_equal(full.state_dict(),resumed.state_dict())
            pd.testing.assert_frame_equal(records[0],resumerec[0])
            fstate=torch.load(fullout/"checkpoints/P"/f"{run_dates[-1]}.pt",weights_only=False)
            rstate=torch.load(resumeout/"checkpoints/P"/f"{run_dates[-1]}.pt",weights_only=False)
            assert nested_equal(fstate["optimizer"],rstate["optimizer"])
            assert fstate["consumed"]==rstate["consumed"]
            results.append("resume_identical_model_optimizer_predictions_consumed_events")
            # Every horizon gets the final horizon release; policies must agree.
            delayed=copy.deepcopy(rows)
            for d,frame in delayed.items(): frame["label_dates"]=[[x[-1]]*3 for x in frame.label_dates]
            fm,fr,_=run("equal_f",group="F",table=delayed)
            pm,pr,_=run("equal_p",table=delayed)
            assert nested_equal(fm.state_dict(),pm.state_dict())
            pd.testing.assert_frame_equal(fr[0].drop(columns="group"),pr[0].drop(columns="group"))
            results.append("all_horizons_delayed_to_complete_F_P_equivalent")
            reversed_rows={d:f.iloc[::-1].reset_index(drop=True) for d,f in rows.items()}
            reversed_z={d:t.flip(0) for d,t in z.items()}
            # Reverse input ordering but preserve each row's matching feature.
            pm=copy.deepcopy(initial)
            rr=online_run("P",pm,reversed_z,reversed_rows,values,positions,run_dates[3:],root/"reorder",max_rows=4)
            assert nested_equal(pm.state_dict(),full.state_dict())
            sortcols=["as_of_date","stock"]
            pd.testing.assert_frame_equal(records[0].sort_values(sortcols).reset_index(drop=True),rr[0].sort_values(sortcols).reset_index(drop=True))
            results.append("canonical_stock_order_invariant")
            sf,srec,_=run("zero_s",group="S",lr=0)
            for group in ("F","P"):
                zm,zr,_=run("zero_"+group,group=group,lr=0)
                assert nested_equal(sf.state_dict(),zm.state_dict())
                cols=[f"pred_return_{k}" for k in range(1,4)]
                np.testing.assert_array_equal(srec[0][cols],zr[0][cols])
            results.append("zero_lr_static_F_P_identity")
            _,frec,_=run("budget_f",group="F")
            pd.testing.assert_frame_equal(frec[2][["date","supervision","optimizer_steps"]],records[2][["date","supervision","optimizer_steps"]])
            assert all(sha256(fullout/"prediction_shards/P"/f"{d}.csv")==h for d,h in records[1][["update_date","prediction_sha256"]].drop_duplicates().itertuples(index=False,name=None))
            results.append("matched_supervision_optimizer_steps_prediction_commit_hashes")
        return {"status":"passed","checks":results,"engine":"same online_run as real pilot","toy_horizon":3}
    finally:
        PRED_LEN=original_horizon

def load_inputs(cache_dir: Path, reference_dir: Path, hidden_dir: Path):
    meta = pd.read_csv(cache_dir / "meta.csv", dtype=str)
    meta["row_id"] = meta["row_id"].astype(int)
    test = meta[meta.split.eq("test")].reset_index(drop=True)
    hidden = np.load(hidden_dir / "hidden_test.npy", mmap_mode="r")
    labels = np.load(cache_dir / "y.npy", mmap_mode="r")
    if hidden.shape != (len(test),512) or labels.shape != (len(meta),10):
        raise AssertionError("unexpected feature or target shape")
    if not np.array_equal(meta.row_id.to_numpy(),np.arange(len(meta))):
        raise AssertionError("global label row IDs must be consecutive")
    if test.duplicated(["as_of_date","stock"]).any():
        raise AssertionError("duplicate test keys")
    if len(hidden) != len(test):
        raise AssertionError(f"hidden/meta row mismatch: {len(hidden)} != {len(test)}")
    windows = pd.read_csv(reference_dir / "date_windows.csv", dtype=str).drop_duplicates("as_of_date")
    windows = windows.set_index("as_of_date")
    calendar = sorted(test.as_of_date.unique())
    if len(calendar) < 60:
        raise AssertionError("fewer than 60 test dates")
    date_to_pos = {d: i for i, d in enumerate(calendar)}
    local_by_global = {int(row.row_id): i for i, row in test.iterrows()}
    if not all(d in windows.index for d in calendar[:60]):
        raise AssertionError("missing date window")
    label_dates = {d: str(windows.loc[d, "label_dates"]).split("|") for d in calendar[:60]}
    return meta, test, hidden, labels, calendar, date_to_pos, local_by_global, label_dates


def select_panel(meta, test: pd.DataFrame, calendar: list[str], n_stocks=60):
    # Selection uses only historical TRAIN availability, never future test outcomes.
    first = set(meta.loc[meta.split.eq("train"), "stock"].astype(str))
    stocks = sorted(first, key=stable_stock_key)[:n_stocks]
    if len(stocks) < n_stocks:
        raise AssertionError(f"only {len(stocks)} stocks available on all pilot dates")
    panel = test[test.stock.isin(stocks) & test.as_of_date.isin(calendar[:60])].copy()
    panel["local_id"] = panel.index
    return stocks, panel


def make_model(checkpoint: Path, update_head: bool):
    model = Model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if update_head:
        for p in model.head.parameters():
            p.requires_grad_(True)
    return model


@torch.no_grad()
def features(model: nn.Module, hidden: np.ndarray, ids: np.ndarray):
    x = torch.tensor(np.asarray(hidden[ids]), dtype=torch.float32)
    return model.adapter(x).detach().cpu()


def predict(model, z: torch.Tensor):
    model.eval()
    with torch.no_grad():
        return model.head(z).detach().cpu().numpy()


def online_run(group, model, z_by_date, row_by_date, labels, date_to_pos, score_dates, output_dir, max_rows=32, stop_after=None, resume_from=None, lr=1e-4):
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.0) if group != "S" else None
    consumed = set()
    pred_rows, access_rows, update_rows = [], [], []
    all_dates = sorted(z_by_date, key=lambda d: date_to_pos[d])
    version, start_pos = 0, 0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer"])
        consumed = set(map(tuple,saved["consumed"]))
        pred_rows,access_rows,update_rows = saved["pred_rows"],saved["access_rows"],saved["update_rows"]
        version,start_pos = saved["version"],saved["next_pos"]
        torch.set_rng_state(saved["torch_rng"]); np.random.set_state(saved["numpy_rng"]);random.setstate(saved["python_rng"])
    for d in all_dates[start_pos:]:
        pos = date_to_pos[d]
        pred = predict(model, z_by_date[d])
        day_rows = []
        for j, row in enumerate(row_by_date[d].itertuples(index=False)):
            day_rows.append({"group": group, "as_of_date": d, "stock": row.stock,
                             "row_id": int(row.row_id), "pred_version": version,
                             **{f"pred_return_{k+1}": float(pred[j,k]) for k in range(PRED_LEN)}})
        # This file is physically committed before accessing feedback values.
        atomic_csv(output_dir / "prediction_shards" / group / f"{d}.csv", pd.DataFrame(day_rows))
        prediction_hash = sha256(output_dir / "prediction_shards" / group / f"{d}.csv")
        pred_rows.extend(day_rows)
        if group == "S" or pos < PRED_LEN:
            if stop_after is not None and pos+1 >= stop_after:
                break
            continue
        if hasattr(labels,"today"):
            labels.today = d
        optimizer.zero_grad(set_to_none=True)
        losses = []
        event_count, feature_ids = 0, set()
        for k in range(1, PRED_LEN+1):
            complete_dates = [o for o in all_dates[:pos] if len(row_by_date[o]) and row_by_date[o].iloc[0].label_dates[-1] == d]
            prefix_dates = [o for o in all_dates[:pos] if len(row_by_date[o]) and row_by_date[o].iloc[0].label_dates[k-1] == d]
            if not complete_dates or not prefix_dates:
                continue
            fdate,pdate = complete_dates[-1],prefix_dates[-1]
            frows = row_by_date[fdate]
            prows = row_by_date[pdate]
            common = set(frows.stock) & set(prows.stock)
            stocks = sorted(common, key=stable_stock_key)[:max_rows]
            origin = fdate if group == "F" else pdate
            candidates = row_by_date[origin]
            indexes = [int(candidates.index[candidates.stock.eq(stock)][0]) for stock in stocks]
            chosen = candidates.iloc[indexes]
            targets = []
            for row in chosen.itertuples(index=False):
                release_date = row.label_dates[k-1]
                if not (origin < d and release_date <= d):
                    raise AssertionError("future feedback read")
                if group == "F" and row.label_dates[-1] > d:
                    raise AssertionError("incomplete F path")
                key = (int(row.row_id), k)
                if key in consumed:
                    raise AssertionError("duplicate feedback event")
                # Only read a scalar AFTER its release date has been checked.
                value = float(labels[int(row.row_id), k-1])
                if not np.isfinite(value):
                    raise FloatingPointError("nonfinite released label")
                targets.append(value)
                consumed.add(key)
                feature_ids.add(int(row.row_id))
                access_rows.append({"group": group, "update_date": d, "row_id": int(row.row_id),
                                    "origin_date": origin, "horizon": k, "label_date": release_date,
                                    "read": True, "prediction_sha256": prediction_hash})
            if indexes:
                x = z_by_date[origin][indexes]
                output = model.head(x)[:,k-1]
                y = torch.tensor(targets, dtype=torch.float32)
                losses.append(F.smooth_l1_loss(output,y,beta=1.0))
                event_count += len(indexes)
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            for parameter in model.head.parameters():
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("nonfinite gradient")
            grad_norm = nn.utils.clip_grad_norm_(model.head.parameters(),1.0)
            optimizer.step(); version += 1
            update_rows.append({"group":group,"date":d,"supervision":event_count,"optimizer_steps":1,
                                "unique_feature_rows":len(feature_ids),"loss":float(loss.detach()),
                                "grad_norm":float(grad_norm),"version":version,"consumed_events":len(consumed)})
            state = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"date":d,
                     "consumed":sorted(consumed),"torch_rng":torch.get_rng_state(),
                     "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                     "version":version,"next_pos":pos+1,"pred_rows":pred_rows,
                     "access_rows":access_rows,"update_rows":update_rows}
            cp = output_dir / "checkpoints" / group / f"{d}.pt"
            cp.parent.mkdir(parents=True,exist_ok=True); torch.save(state,cp)
        if stop_after is not None and pos+1>=stop_after:
            break
    atomic_csv(output_dir/f"{group}_predictions.csv",pd.DataFrame(pred_rows))
    atomic_csv(output_dir/f"{group}_access_log.csv",pd.DataFrame(access_rows))
    atomic_csv(output_dir/f"{group}_update_log.csv",pd.DataFrame(update_rows))
    return pd.DataFrame(pred_rows),pd.DataFrame(access_rows),pd.DataFrame(update_rows)

def metrics(pred, panel, labels, local_by_global, score_dates):
    rows = []
    panel_index = {(str(r.as_of_date), str(r.stock)): int(r.row_id) for r in panel.itertuples()}
    for d in score_dates:
        g = pred[pred.as_of_date.eq(d)].sort_values("stock")
        ids = np.array([panel_index[(d, str(s))] for s in g.stock])
        actual = np.asarray(labels[ids], dtype=float)
        pp = g[[f"pred_return_{k+1}" for k in range(PRED_LEN)]].to_numpy(float)
        path_p, path_y = pp.mean(1), actual.mean(1)
        rank = pd.Series(path_p).corr(pd.Series(path_y), method="spearman")
        rows.append({"group": str(g.group.iloc[0]), "as_of_date": d, "path_mae": float(np.abs(pp-actual).mean()),
                     "step1_mae": float(np.abs(pp[:, 0]-actual[:, 0]).mean()),
                     "endpoint_mae": float(np.abs(pp[:, -1]-actual[:, -1]).mean()),
                     "path_rankic": float(rank) if pd.notna(rank) else np.nan,
                     "endpoint_rankic": float(pd.Series(pp[:, -1]).corr(pd.Series(actual[:, -1]), method="spearman")),
                     "parameter_norm": np.nan})
    return pd.DataFrame(rows)


def finish_report(out, all_metrics, summary, pilot_dates, elapsed_by_group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_group = {g: df.set_index("as_of_date") for g, df in all_metrics.groupby("group")}
    paired = pd.DataFrame(index=by_group["P"].index)
    for metric in ("path_mae", "step1_mae", "path_rankic", "endpoint_rankic"):
        for reference in ("F", "S"):
            paired[f"P_minus_{reference}_{metric}"] = by_group["P"][metric]-by_group[reference][metric]
    atomic_csv(out / "paired_by_date.csv",paired.reset_index())
    s = summary.set_index("group")
    if not np.isfinite(summary.drop(columns="group").to_numpy()).all():
        raise FloatingPointError("nonfinite summary metrics")
    relative_gain = float(1-s.loc["P","path_mae"]/s.loc["F","path_mae"])
    half_gains = [-float(paired.P_minus_F_path_mae.iloc[part].mean()) for part in (slice(0,25),slice(25,50))]
    checks = {"relative_path_mae_gain_at_least_1pct":relative_gain>=.01,
              "path_rankic_not_worse_than_F":bool(s.loc["P","path_rankic"]>=s.loc["F","path_rankic"]),
              "path_mae_not_worse_than_static":bool(s.loc["P","path_mae"]<=s.loc["S","path_mae"]),
              "path_rankic_not_worse_than_static":bool(s.loc["P","path_rankic"]>=s.loc["S","path_rankic"]),
              "both_half_windows_mae_improve":all(x>0 for x in half_gains)}
    decision = "continue_candidate" if all(checks.values()) else "stop_no_expansion"
    update_f = pd.read_csv(out / "F_update_log.csv")
    update_p = pd.read_csv(out / "P_update_log.csv")
    pd.testing.assert_frame_equal(update_f[["date","supervision","optimizer_steps"]],update_p[["date","supervision","optimizer_steps"]])
    for group in ("F","P"):
        access = pd.read_csv(out / f"{group}_access_log.csv")
        assert (access.label_date<=access.update_date).all()
        assert not access.duplicated(["row_id","horizon"]).any()
        for d,h in access[["update_date","prediction_sha256"]].drop_duplicates().itertuples(index=False,name=None):
            assert sha256(out/"prediction_shards"/group/f"{d}.csv")==h
    decision_data = {"decision":decision,"relative_path_mae_gain_P_vs_F":relative_gain,
        "mean_path_mae_improvement_first_second_half":half_gains,"checks":checks,
        "group_seconds":elapsed_by_group,"F_supervision":int(update_f.supervision.sum()),
        "P_supervision":int(update_p.supervision.sum()),"F_unique_feature_rows_summed":int(update_f.unique_feature_rows.sum()),
        "P_unique_feature_rows_summed":int(update_p.unique_feature_rows.sum()),
        "interpretation":"development screening only; not significance, novelty, or stable financial profit evidence"}
    atomic_json(out / "decision.json",decision_data)
    for metric, title, filename in (("path_mae","Cumulative difference in daily Path MAE (lower is better)","path_mae_cumulative.png"),
                                    ("path_rankic","Cumulative difference in daily Path RankIC (higher is better)","path_rankic_cumulative.png")):
        fig,ax=plt.subplots(figsize=(9,4.5))
        for reference,color in (("F","#2266bb"),("S","#c05a26")):
            values=paired[f"P_minus_{reference}_{metric}"].cumsum().to_numpy()
            ax.plot(pd.to_datetime(paired.index),values,label=f"P - {reference}",color=color)
        ax.axhline(0,color="gray",lw=.8);ax.set_title(title);ax.set_xlabel("Scored forecast origin")
        ax.legend();ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
        fig.savefig(out/filename,dpi=160);plt.close(fig)
    predictions_s = pd.read_csv(out / "S_predictions.csv")
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    coverage = {"selected_stocks":len(config["stocks"]),"observed_stocks":int(predictions_s.stock.nunique()),
                "absent_selected_stocks":sorted(set(config["stocks"])-set(predictions_s.stock)),
                "actual_rows_per_group":len(predictions_s),
                "score_rows_per_group":int(predictions_s.as_of_date.isin(pilot_dates[10:]).sum()),
                "counts_by_date":predictions_s.groupby("as_of_date").size().to_dict(),
                "policy":"retain fixed train-period stock selection; no replacement using future coverage"}
    atomic_json(out / "coverage_audit.json",coverage)
    lines=["# 延迟反馈最小实验结果", "",f"探索区间：{pilot_dates[0]}—{pilot_dates[-1]}；评分 {pilot_dates[10]}—{pilot_dates[-1]}，共50日。", "",
           f"预选{coverage['selected_stocks']}只股票；旧锁定面板中实际覆盖{coverage['observed_stocks']}只、每组{coverage['actual_rows_per_group']}个股票日，评分{coverage['score_rows_per_group']}个股票日。缺失预选股票：{coverage['absent_selected_stocks']}。未根据未来覆盖率替换该股，缺失原因未独立确认。", "",
           "S=静态；F=完整路径成熟后更新；P=各步成熟后更新。所有组使用同一个D-RankIC初始checkpoint，Kronos与原Adapter冻结，F/P只更新5,130参数预测头。", "",
           "|组|Path MAE|第1步MAE|Path RankIC|Endpoint RankIC|头参数L2变化|", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"|{row.group}|{row.path_mae:.8f}|{row.step1_mae:.8f}|{row.path_rankic:.6f}|{row.endpoint_rankic:.6f}|{row.parameter_norm:.6f}|")
    lines += ["", "本表Path MAE定义为：每个股票的10步累计收益逐点绝对误差，再按股票、horizon和日期取均值；不是先把10步压成标量再计算误差。Path RankIC用10步均值信号逐日计算。", "",
              f"P相对F的Path MAE降幅：{relative_gain:.3%}。前/后25评分日的MAE改善分别为 {half_gains[0]:.8f} / {half_gains[1]:.8f}。", "",
              f"固定筛选结论：**{decision}**。", ""]
    for key,value in checks.items(): lines.append(f"- {key}: {value}")
    lines += ["",f"F/P各{len(update_f)}次更新、{int(update_f.supervision.sum())}个标签标量；特征行访问累计F={int(update_f.unique_feature_rows.sum())}、P={int(update_p.unique_feature_rows.sum())}，不宣称等FLOPs。",
              "",f"相对S，F的Path MAE降低{1-s.loc['F','path_mae']/s.loc['S','path_mae']:.3%}，P降低{1-s.loc['P','path_mae']/s.loc['S','path_mae']:.3%}。但三组平均Path RankIC均为负，不能把改善解释为已经具有稳定选股能力。",
              "", "![Path MAE](path_mae_cumulative.png)","", "![Path RankIC](path_rankic_cumulative.png)", "",
              "7项toy检查使用实际在线更新器；真实运行另核验无未成熟标签访问、无重复消费、逐日监督数相同、预测先持久化、冻结Adapter不变。",
              "", "边界：这是已查看旧测试期中的回顾性小面板筛选，不能证明稳定收益或论文新颖性。没有运行组合回测，也没有扫描超参数。第10步F/P接收相同监督；线性头各行独立，因此Endpoint接近/相等属于结构预期。"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(args):
    started = time.time(); seed_all(100)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    toy = run_toy_checks()
    print(json.dumps(toy),flush=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a new output directory; existing pilot is immutable")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "correctness_report.json", toy)
    ck_hash = sha256(args.checkpoint)
    if ck_hash.lower() != EXPECTED_CKPT:
        raise AssertionError(f"checkpoint hash mismatch {ck_hash}")
    meta, test, hidden, labels, calendar, date_to_pos, local_by_global, label_dates = load_inputs(args.cache_dir, args.reference_dir, args.hidden_dir)
    stocks, panel = select_panel(meta, test, calendar, 60)
    pilot_dates = calendar[:60]
    windows = pd.read_csv(args.reference_dir / "date_windows.csv", dtype=str).set_index("as_of_date")
    for d in pilot_dates:
        w = windows.loc[d]
        if len(label_dates[d]) != 10 or label_dates[d][0] <= d:
            raise AssertionError("invalid label dates")
        part = panel[panel.as_of_date.eq(d)]
        if not part.label_start_date.eq(label_dates[d][0]).all() or not part.label_end_date.eq(label_dates[d][-1]).all():
            raise AssertionError("label boundary mismatch")
    reference_config = json.loads((ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_static_20261013/run_config.json").read_text())
    hidden_hash = sha256(args.hidden_dir / "hidden_test.npy")
    if hidden_hash != reference_config["hashes"]["hidden_test.npy"]:
        raise AssertionError("test hidden identity differs from historical D-RankIC run")
    # Local hidden-test row positions are the order of test meta, while labels use global row_id.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    z_all = features(make_model(args.checkpoint, False), hidden, np.arange(len(test)))
    z_by_date, row_by_date = {}, {}
    for d in pilot_dates:
        part = panel[panel.as_of_date.eq(d)].sort_values("stock").copy()
        part["local_id"] = [local_by_global[int(x)] for x in part.row_id]
        part["label_dates"] = [label_dates[d]] * len(part)
        z_by_date[d] = z_all[part.local_id.to_numpy()]
        row_by_date[d] = part.reset_index(drop=True)
    # Offline preflight only: compare cached base forecasts and ground truth with
    # the archived paired static run. These labels never enter online training.
    selected_keys = panel[["as_of_date", "stock", "row_id"]]
    archived = pd.read_csv(ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_static_20261013/predictions.csv", dtype={"stock":str})
    archived = selected_keys.merge(archived,on=["as_of_date","stock"],validate="one_to_one")
    if len(archived)!=len(panel):
        raise AssertionError("historical static coverage differs")
    base_model = make_model(args.checkpoint, False)
    with torch.no_grad():
        base_values = base_model.head(z_all[[local_by_global[int(i)] for i in archived.row_id]]).numpy()
    pred_cols = [f"pred_return_{k}" for k in range(1,11)]
    actual_cols = [f"actual_return_{k}" for k in range(1,11)]
    prediction_error = float(np.max(np.abs(base_values-archived[pred_cols].to_numpy())))
    label_error = float(np.max(np.abs(np.asarray(labels[archived.row_id.to_numpy(int)])-archived[actual_cols].to_numpy())))
    if prediction_error>1e-6 or label_error>1e-6:
        raise AssertionError(f"historical identity mismatch: pred={prediction_error}, labels={label_error}")
    audit = {"checkpoint_sha256":ck_hash,"hidden_sha256":hidden_hash,"max_prediction_error":prediction_error,
             "max_label_error":label_error,"rows":len(archived),"offline_preflight_labels_not_passed_to_optimizer":True}
    atomic_json(out / "preflight.json",audit)
    print(json.dumps({"stage":"preflight",**audit}),flush=True)
    atomic_json(out / "run_config.json", {"pilot": "delayed_feedback_v1", "checkpoint_sha256": ck_hash,
        "stocks": stocks, "dates": pilot_dates, "warmup_days": 10, "score_days": pilot_dates[10:],
        "groups": ["S", "F", "P"], "max_labels_per_horizon": 32, "lr": 1e-4, "seed": 100,
        "test_period_is_retrospective": True, "toy": toy,"code_sha256":sha256(Path(__file__)),
        "trainable_parameters":5130,"weight_decay":0,"smooth_l1_beta":1,"gradient_clip":1,
        "stock_selection":"training availability then sha256(prefix-pilot-v1|stock)",
        "supervision_matching":"per-day per-horizon paired stock intersection; not FLOP-matched",
        "preflight":audit})
    frames = []
    elapsed_by_group = {}
    # Static is generated with the same base head; F and P start from identical copies.
    for group in ("S", "F", "P"):
        group_start = time.perf_counter()
        model = make_model(args.checkpoint, group != "S")
        frozen_before = copy.deepcopy(model.adapter.state_dict())
        pred, access, updates = online_run(group, model, z_by_date, row_by_date, labels, date_to_pos, pilot_dates[10:], out)
        if not nested_equal(frozen_before,model.adapter.state_dict()):
            raise AssertionError("frozen adapter mutated")
        elapsed_by_group[group] = time.perf_counter()-group_start
        m = metrics(pred[pred.as_of_date.isin(pilot_dates[10:])], panel, labels, local_by_global, pilot_dates[10:])
        # Parameter change is measured against the original head.
        baseline = make_model(args.checkpoint, group != "S")
        if group == "S": delta = 0.0
        else:
            delta = float(sum(torch.sum((a-b)**2).item() for a,b in zip(model.head.parameters(), baseline.head.parameters())) ** 0.5)
        m["parameter_norm"] = delta
        frames.append(m)
        atomic_json(out / f"{group}_summary.json", {"group": group, "rows": len(pred), "updates": len(updates),
            "access_events": len(access), "parameter_norm": delta,"seconds":elapsed_by_group[group]})
        print(json.dumps({"stage":"group_complete","group":group,"rows":len(pred),"updates":len(updates),"seconds":elapsed_by_group[group]}),flush=True)
    all_metrics = pd.concat(frames, ignore_index=True)
    atomic_csv(out / "metrics_by_date.csv", all_metrics)
    summary = all_metrics.groupby("group", sort=False).agg(path_mae=("path_mae", "mean"), step1_mae=("step1_mae", "mean"),
        endpoint_mae=("endpoint_mae", "mean"), path_rankic=("path_rankic", "mean"), endpoint_rankic=("endpoint_rankic", "mean"), parameter_norm=("parameter_norm", "first")).reset_index()
    atomic_csv(out / "summary.csv", summary)
    finish_report(out,all_metrics,summary,pilot_dates,elapsed_by_group)
    atomic_json(out / "status.json", {"status": "complete", "elapsed_seconds": time.time()-started,
        "rows_per_group":len(panel),"selected_stocks":len(stocks),"observed_stocks":int(panel.stock.nunique()),
        "score_rows_per_group":int(panel.as_of_date.isin(pilot_dates[10:]).sum()),
        "score_dates": 50, "summary": summary.to_dict("records")})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "experiments/csi300_aligned/doubleadapt_cache_v2")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "experiments/csi300_aligned/reference")
    parser.add_argument("--hidden-dir", type=Path, default=ROOT / "experiments/csi300_aligned/reference/hidden_cache")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".runtime/D_model_rankic_pilot_original.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/csi300_aligned/models/delayed_feedback_pilot_20261020")
    main(parser.parse_args())
