"""Retrospective 226-day HCA experiment; the original A stream engine is unchanged."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.experiments import run_adapter_online_full as full

pilot = full.pilot
old = pilot.old
KEYS, PCOLS, ACOLS = full.KEYS, full.PCOLS, full.ACOLS
VARIANTS = ("Static", "A", "HCA_PCA", "HCA_Random")
HISTORY = ROOT / "experiments/csi300_aligned/models/adapter_online_full_20261020"
DEFAULT_OUT = ROOT / "experiments/csi300_aligned/models/hca_online_full_20261020"


def canonical_columns(basis):
    """Fix eigensolver sign ambiguity without consulting evaluation data."""
    basis = basis.copy()
    for j in range(basis.shape[1]):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def training_basis(meta, values, rank=2):
    train = meta.loc[meta.split.eq("train")]
    ids = train.row_id.to_numpy(dtype=int)
    assert len(ids) > rank and not train.row_id.duplicated().any()
    # Index BEFORE conversion: validation/test values are never used to fit B.
    y = np.asarray(values[ids], dtype=np.float64)
    assert np.isfinite(y).all() and y.shape[1] > rank
    center = y.mean(0)
    centered = y - center
    covariance = centered.T @ centered / len(y)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = canonical_columns(vectors[:, order[:rank]]).astype(np.float32)
    return basis, {"fit_split": "train", "rows": len(ids), "rank": rank,
        "origin_start": str(train.as_of_date.min()), "origin_end": str(train.as_of_date.max()),
        "label_end": str(train.label_end_date.max()) if "label_end_date" in train else None,
        "center": center.tolist(), "covariance_divisor": len(y),
        "explained_variance_ratio": (eigenvalues / eigenvalues.sum()).tolist(),
        "retained_variance": float(eigenvalues[:rank].sum() / eigenvalues.sum())}


def random_basis(horizon=10, rank=2):
    q, _ = np.linalg.qr(np.random.default_rng(20261020).normal(size=(horizon, rank)))
    return canonical_columns(q).astype(np.float32)


class ResidualHead(nn.Module):
    def __init__(self, base, basis):
        super().__init__()
        self.base = base
        self.coeff = nn.Linear(base.in_features, basis.shape[1])
        nn.init.zeros_(self.coeff.weight)
        nn.init.zeros_(self.coeff.bias)
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32).clone())
        assert self.basis.shape[0] == base.out_features

    def forward(self, z):
        return self.base(z) + self.coeff(z) @ self.basis.T


def with_hca(initial, basis):
    model = copy.deepcopy(initial)
    model.head = ResidualHead(model.head, basis)
    model.eval()
    return model


def parts(model):
    if isinstance(model.head, ResidualHead):
        return {"adapter": model.adapter, "head": model.head.base, "coeff": model.head.coeff}
    return {"adapter": model.adapter, "head": model.head}


def parameter_changes(initial, model):
    left, right = parts(initial), parts(model)
    return {name: math.sqrt(sum(float(((p.detach() - q.detach()) ** 2).sum())
        for p, q in zip(left[name].parameters(), right[name].parameters()))) for name in left}


def instrumented_run(model, *args, **kwargs):
    """Leaf hooks observe pre-clipping gradients; no engine/optimizer mutation."""
    gradients, handles = {}, []
    for name, module in parts(model).items():
        for parameter_name, p in module.named_parameters():
            key = name + "." + parameter_name
            gradients[key] = []
            def observe(gradient, key=key):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite gradient " + key)
                gradients[key].append(float(gradient.detach().square().sum()))
            # configure() in run_stream sets requires_grad after hooks are attached.
            p.requires_grad_(True)
            handles.append(p.register_hook(observe))
    try:
        result = pilot.run_stream("A", model, *args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    updates = result[2].copy()
    for name in parts(model):
        samples = [v for key, v in gradients.items() if key.startswith(name + ".")]
        assert all(len(v) == len(updates) for v in samples)
        updates[name + "_grad_norm"] = np.sqrt(np.sum(samples, axis=0))
    result[2] = updates
    return result


def correctness_checks():
    original = pilot.correctness_checks()
    old.seed_all(100)
    dates = [f"2020-01-{i + 1:02d}" for i in range(20)]
    run_dates = dates[:14]
    initial = pilot.ToyModel().eval()
    hh = {d: torch.randn(5, 4) for d in run_dates}
    with torch.no_grad():
        zz = {d: initial.adapter(h) for d, h in hh.items()}
    rows = {d: pd.DataFrame([{"as_of_date": d, "stock": f"s{i}", "row_id": j * 5 + i,
        "label_dates": dates[j+1:j+4]} for i in range(5)]) for j, d in enumerate(run_dates)}
    releases = {int(r.row_id): r.label_dates for f in rows.values() for r in f.itertuples()}
    values = np.random.default_rng(100).normal(0, .01, (70, 3))
    schedule = pilot.make_schedule(run_dates, rows, 3, 4)
    meta = pd.DataFrame({"split": ["train"]*40+["val"]*10+["test"]*20,
        "row_id": np.arange(70), "as_of_date": ["2019-01-01"]*70})
    b, _ = training_basis(meta, values)
    poisoned_fit = values.copy(); poisoned_fit[40:] = np.nan
    np.testing.assert_array_equal(b, training_basis(meta, poisoned_fit)[0])
    checks = ["basis_fit_ignores_all_val_test_targets"]
    for basis in (b, random_basis(3)):
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6, rtol=0)
    with tempfile.TemporaryDirectory(prefix="hca_checks_") as task_dir:
        root = Path(task_dir)
        def run(name, basis=None, data=values, **kwargs):
            model = copy.deepcopy(initial) if basis is None else with_hca(initial, basis)
            result = instrumented_run(model, hh, zz, rows, schedule,
                pilot.ReleasedLabels(data, releases), root / name, **kwargs)
            return model, result
        base, baseline = run("base")
        zero, record = run("zero_basis", np.zeros((3, 2), dtype=np.float32))
        np.testing.assert_array_equal(record[0].filter(regex="^pred_return_"), baseline[0].filter(regex="^pred_return_"))
        for name in ("adapter", "head"):
            assert old.nested_equal(parts(base)[name].state_dict(), parts(zero)[name].state_dict())
        checks.append("zero_basis_exactly_reduces_to_A_parameters_and_predictions")
        for kind, basis in (("PCA", b), ("Random", random_basis(3))):
            start = with_hca(initial, basis)
            for p in start.head.coeff.parameters():
                assert torch.count_nonzero(p) == 0
            with torch.no_grad():
                for d in run_dates:
                    torch.testing.assert_close(start(hh[d]), initial(hh[d]), atol=0, rtol=0)
            model, rec = run(kind, basis)
            assert torch.equal(model.head.basis, torch.tensor(basis))
            assert not model.head.basis.requires_grad
            assert (rec[2].coeff_grad_norm > 0).all()
            assert all(v > 0 for v in parameter_changes(start, model).values())
            pd.testing.assert_frame_equal(rec[1].drop(columns="prediction_sha256"), baseline[1].drop(columns="prediction_sha256"))
            cutoff = run_dates[7]
            poison = values.copy()
            for rid, dd in releases.items():
                for k, release in enumerate(dd):
                    if release > cutoff: poison[rid, k] = np.nan
            clean, cr = run(kind+"_cut", basis, stop_after=8)
            bad, br = run(kind+"_poison", basis, poison, stop_after=8)
            assert old.nested_equal(clean.state_dict(), bad.state_dict())
            for a, c in zip(cr, br): pd.testing.assert_frame_equal(a, c)
            cp = root / (kind+"_cut") / "checkpoints/A" / f"{cutoff}.pt"
            bp = root / (kind+"_poison") / "checkpoints/A" / f"{cutoff}.pt"
            assert old.nested_equal(torch.load(cp, weights_only=False)["optimizer"], torch.load(bp, weights_only=False)["optimizer"])
            # Resume uses the same engine; gradient diagnostics join committed prefix below.
            resumed = with_hca(initial, basis)
            rr = pilot.run_stream("A", resumed, hh, zz, rows, schedule,
                pilot.ReleasedLabels(values, releases), root / (kind+"_cut"), resume=cp)
            assert old.nested_equal(model.state_dict(), resumed.state_dict())
            for a, c in zip(rec[:2], rr[:2]): pd.testing.assert_frame_equal(a, c)
            for key in ("model", "optimizer", "consumed"):
                final = torch.load(root/kind/"checkpoints/A"/f"{run_dates[-1]}.pt", weights_only=False)
                again = torch.load(root/(kind+"_cut")/"checkpoints/A"/f"{run_dates[-1]}.pt", weights_only=False)
                assert old.nested_equal(final[key], again[key])
            versions = rec[0].groupby("as_of_date").pred_version.first().to_numpy()
            np.testing.assert_array_equal(versions, np.maximum(0, np.arange(14)-3))
            for d, digest in rec[1][["update_date", "prediction_sha256"]].drop_duplicates().itertuples(index=False, name=None):
                assert old.sha256(root/kind/"prediction_shards/A"/f"{d}.csv") == digest
        checks += ["both_HCA_initial_forecasts_identical_basis_frozen_coeff_learns",
            "both_HCA_future_poison_predictions_parameters_logs_optimizer_identical",
            "both_HCA_resume_models_optimizer_consumed_predictions_identical",
            "both_HCA_matched_events_prediction_commit_and_next_day_effect"]
    return {"status": "passed", "original_engine_checks": original, "hca_checks": checks}


def audit_stream(out, name, result, initial, model, dates, reference, historical):
    prediction, access, updates, _ = result
    ordered = prediction.sort_values(KEYS).reset_index(drop=True)
    assert ordered[KEYS].equals(reference[KEYS])
    assert np.isfinite(ordered[PCOLS].to_numpy()).all() and not ordered.duplicated(KEYS).any()
    is_static = name.endswith("_Static")
    error = None
    if is_static or name.endswith("_A"):
        expected = pd.read_csv(historical, dtype={"stock":str}).sort_values(KEYS).reset_index(drop=True)
        assert expected[KEYS].equals(ordered[KEYS])
        error = float(np.abs(expected[PCOLS].to_numpy()-ordered[PCOLS].to_numpy()).max())
        assert error <= 1e-7, (name, error)
    changes = parameter_changes(initial, model)
    if is_static:
        assert all(v == 0 for v in changes.values()) and access.empty and updates.empty
    else:
        assert len(updates) == 216 and len(access) == 69120
        assert updates.date.tolist() == dates[10:]
        assert (access.label_date <= access.update_date).all() and (access.complete_date <= access.update_date).all()
        assert not access.duplicated(["row_id", "horizon"]).any()
        assert updates.supervision.eq(320).all() and updates.optimizer_steps.eq(1).all()
        assert all(v > 0 for v in changes.values())
        for part in parts(model): assert (updates[part+"_grad_norm"] > 0).all()
        for d, digest in access[["update_date", "prediction_sha256"]].drop_duplicates().itertuples(index=False, name=None):
            assert old.sha256(out/"streams"/name/"prediction_shards/A"/f"{d}.csv") == digest
        versions = prediction.groupby("as_of_date").pred_version.agg(["min", "max"])
        for col in ("min", "max"):
            np.testing.assert_array_equal(versions[col], np.maximum(0, np.arange(226)-10))
    basis_unchanged = None
    if isinstance(model.head, ResidualHead):
        assert torch.equal(initial.head.basis, model.head.basis)
        assert not model.head.basis.requires_grad
        basis_unchanged = True
    return {"stream":name, "rows":len(ordered), "updates":len(updates), "supervision":len(access),
        "historical_max_abs_error":error, "parameter_delta_l2":changes, "basis_unchanged":basis_unchanged,
        "future_reads":0, "duplicate_consumption":0, "status":"passed"}


def make_report(out, names, dates):
    daily, rows, half_rows, pairs, bootstrap = {}, [], [], [], []
    comparisons = [(f"{c}_{a}",f"{c}_{b}") for c in full.MODELS for a,b in
        (("A","Static"),("HCA_PCA","A"),("HCA_Random","A"),("HCA_PCA","HCA_Random"))]
    for window, dd in (("full_226",dates),("post_warmup_216",dates[10:])):
        for name in names:
            directory = out/"evaluation"/window/(name+"_predictions")
            summary = json.loads((directory/"evaluation_summary.json").read_text(encoding="utf-8"))
            for signal in ("path","endpoint"):
                cross = pd.read_csv(directory/f"evaluation_{signal}_cross_section_daily.csv")
                port = pd.read_csv(directory/f"evaluation_{signal}_portfolio_daily.csv")
                f = cross.merge(port,on="as_of_date",validate="one_to_one",suffixes=("","_portfolio")).sort_values("as_of_date").reset_index(drop=True)
                assert f.as_of_date.tolist() == dd
                assert np.isfinite(f[["ic","rankic","top_net"]].to_numpy()).all()
                np.testing.assert_allclose(f.top_net, f.top_raw-.0015*f.turnover,atol=1e-14,rtol=0)
                daily[window,name,signal] = f
                row = {"window":window,"model":name,"signal":signal,"dates":len(f),
                    "ic":float(f.ic.mean()),"rankic":float(f.rankic.mean()),"turnover":float(f.turnover.mean()),
                    **{k:float(v) for k,v in full.return_stats(f.top_net.to_numpy()).items()},
                    "trajectory_mae":summary["metrics"]["trajectory_pooled"]["mae"],
                    "trajectory_rmse":summary["metrics"]["trajectory_pooled"]["rmse"]}
                rows.append(row)
                if window == "post_warmup_216":
                    for half, part in (("first108",f.iloc[:108]),("last108",f.iloc[108:])):
                        half_rows.append({"model":name,"signal":signal,"half":half,"dates":len(part),
                            "start":part.as_of_date.iloc[0],"end":part.as_of_date.iloc[-1],
                            "ic":float(part.ic.mean()),"rankic":float(part.rankic.mean()),
                            **{k:float(v) for k,v in full.return_stats(part.top_net.to_numpy()).items()}})
        draws = full.indices(len(dd),5000,10,20261013)
        for a,b in comparisons:
            for signal in ("path","endpoint"):
                aa,bb = daily[window,a,signal],daily[window,b,signal]
                diff = pd.DataFrame({"window":window,"comparison":a+"_minus_"+b,"signal":signal,"as_of_date":aa.as_of_date})
                for metric in ("ic","rankic","top_net","turnover","top_raw","transaction_cost"):
                    delta = aa[metric].to_numpy()-bb[metric].to_numpy()
                    diff[metric+"_delta"] = delta
                    boot = delta[draws].mean(1)
                    bootstrap.append({"window":window,"comparison":a+"_minus_"+b,"signal":signal,"metric":"mean_"+metric,
                        "delta":float(delta.mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))})
                pairs.append(diff)
                for metric,value in full.return_stats(aa.top_net.to_numpy()).items():
                    delta = value-full.return_stats(bb.top_net.to_numpy())[metric]
                    boot = full.return_stats(aa.top_net.to_numpy()[draws])[metric]-full.return_stats(bb.top_net.to_numpy()[draws])[metric]
                    bootstrap.append({"window":window,"comparison":a+"_minus_"+b,"signal":signal,"metric":metric,
                        "delta":float(delta),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))})
    table = pd.DataFrame(rows); halves = pd.DataFrame(half_rows)
    old.atomic_csv(out/"comparison_metrics.csv",table)
    old.atomic_csv(out/"half_metrics.csv",halves)
    old.atomic_csv(out/"paired_daily_differences.csv",pd.concat(pairs,ignore_index=True))
    old.atomic_csv(out/"paired_block10_bootstrap.csv",pd.DataFrame(bootstrap))
    primary = table[table.window.eq("full_226")].set_index(["model","signal"])
    half = halves.set_index(["model","signal","half"])
    decisions = {}
    for checkpoint in full.MODELS:
        a = primary.loc[(checkpoint+"_A","path")]
        ae = primary.loc[(checkpoint+"_A","endpoint")]
        random = primary.loc[(checkpoint+"_HCA_Random","path")]
        for variant in ("HCA_PCA","HCA_Random"):
            name = checkpoint+"_"+variant
            p,e = primary.loc[(name,"path")],primary.loc[(name,"endpoint")]
            checks = {"path_return_improves":bool(p.net_cumulative_return>a.net_cumulative_return),
                "path_ic_not_lower":bool(p.ic>=a.ic),
                # Conservative predeclared interpretation of 'no obvious endpoint reversal'.
                "endpoint_return_and_ic_not_lower":bool(e.net_cumulative_return>=ae.net_cumulative_return and e.ic>=ae.ic),
                "drawdown_within_5pp":bool(p.max_drawdown>=a.max_drawdown-.05)}
            checks["both108_halves_return_or_ic_improves"] = all(
                half.loc[(name,"path",h)].net_cumulative_return>half.loc[(checkpoint+"_A","path",h)].net_cumulative_return
                or half.loc[(name,"path",h)].ic>half.loc[(checkpoint+"_A","path",h)].ic for h in ("first108","last108"))
            if variant == "HCA_PCA":
                checks["pca_return_and_ic_at_least_random_one_strict"] = bool(
                    p.net_cumulative_return>=random.net_cumulative_return and p.ic>=random.ic and
                    (p.net_cumulative_return>random.net_cumulative_return or p.ic>random.ic))
            positive = all(checks.values())
            conflict = (p.net_cumulative_return>a.net_cumulative_return) != (p.ic>=a.ic)
            decision = "positive_candidate" if positive else ("partial_metric_conflict" if conflict else "no_consistent_positive_optimization")
            decisions[name] = {"decision":decision,"checks":checks,
                "path_return_delta":float(p.net_cumulative_return-a.net_cumulative_return),"path_ic_delta":float(p.ic-a.ic)}
    old.atomic_json(out/"decisions.json",decisions)
    old.atomic_json(out/"evaluation_summary.json",{"retrospective":True,"metrics":rows,"decisions":decisions,
        "half_window":"post-warmup216 independently evaluated cash-start portfolio sliced into108+108; no half resets",
        "bootstrap":"paired realized daily returns, not retraining or resimulating holdings; no multiple comparison adjustment"})
    old.atomic_csv(out/"evaluation_summary.csv",table)
    full.draw_plots(out,names,daily)
    lines = ["# HCA 完整226日回顾性实验", "", "2025-07-01—2026-06-05，66,103个股票日期键；全部模型同窗口、同成本、同持仓规则。",
        "Kronos冻结，Adapter/head按A联合训练，HCA新增零初始化Linear(512,2)及冻结basis；只用完整成熟标签。", "",
        "|模型|信号|累计净收益|IC|RankIC|Sharpe|最大回撤|换手率|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (name,signal),r in primary.iterrows():
        lines.append(f"|{name}|{signal}|{r.net_cumulative_return:.3%}|{r.ic:.6f}|{r.rankic:.6f}|{r.sharpe:.3f}|{r.max_drawdown:.3%}|{r.turnover:.3%}|")
    lines += ["", "|对比A|Path收益差(百分点)|Path IC差|判定|", "|---|---:|---:|---|"]
    for name,d in decisions.items():
        lines.append(f"|{name}|{d['path_return_delta']*100:+.3f}|{d['path_ic_delta']:+.6f}|{d['decision']}|")
        lines.append("")
        lines.append(f"{name} checks: `{json.dumps(d['checks'])}`")
    lines += ["", "## 配对区间", "", "|对比|信号|指标|差值|95%下限|95%上限|", "|---|---|---|---:|---:|---:|"]
    for r in bootstrap:
        if r["window"]=="full_226" and "HCA" in r["comparison"] and r["metric"] in ("mean_ic","net_cumulative_return"):
            lines.append(f"|{r['comparison']}|{r['signal']}|{r['metric']}|{r['delta']:+.6f}|{r['ci_low']:+.6f}|{r['ci_high']:+.6f}|")
    lines += ["", "evaluate：Top30/Drop3、最低持有5日、单向15bps、actual_return_1日收益、年化252；IC为逐日截面Pearson均值。",
        "half_metrics.csv为216日辅助组合持仓收益按前108/后108日切片，不重新建仓；完整226日主结果包含10日共同暖启动。",
        "bootstrap block10、5000次，共享日期块，重采样已实现日收益，不重训练/回放持仓，未做多重比较校正。",
        "Endpoint守卫保守定义为收益和IC均不下降；PCA结构守卫定义为Path收益/IC均不低于Random且至少一项严格更高。",
        "这是已经查看的固定CSI300面板的回顾性验证，保留成分股前视/幸存者偏差；不能作为独立确认、稳定收益或论文最终证据。",
        "只验证complete-label结构化残差适配，不证明短期标签提前改善长期预测。失败后不叠加MOG/SVA。", "",
        "![累计净收益](cumulative_net_returns.png)", "", "![20日均值IC](ic_curves.png)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(args):
    started = time.perf_counter()
    checks = correctness_checks()
    print(json.dumps(checks),flush=True)
    if args.checks_only: return
    old.seed_all(100); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError("fresh output directory required")
    meta,test,hidden,values,dates,stocks,rows,hh,schedule,releases,hidden_hash = full.load_data()
    basis,basis_info = training_basis(meta,values)
    assert basis_info["rows"]==525700 and basis_info["label_end"]<dates[0]
    out.mkdir(parents=True,exist_ok=True)
    old.atomic_json(out/"status.json",{"status":"running"})
    old.atomic_json(out/"correctness_report.json",checks)
    for kind,array in (("pca",basis),("random",random_basis())):
        np.testing.assert_allclose(array.T@array,np.eye(2),atol=1e-6,rtol=0)
        np.save(out/f"basis_{kind}.npy",array)
    old.atomic_json(out/"basis_manifest.json",{"pca":basis_info,"random_seed":20261020,
        "matrices":{kind:{"values":np.load(out/f"basis_{kind}.npy").tolist(),"sha256":old.sha256(out/f"basis_{kind}.npy")}
            for kind in ("pca","random")},"fit_targets":"train cumulative returns only; centered covariance / n; deterministic column signs"})
    inputs = {"runner":Path(__file__),"full_runner":Path(full.__file__),"engine":Path(pilot.__file__),"original_engine":Path(old.__file__),
        "meta":ROOT/"experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv",
        "labels":ROOT/"experiments/csi300_aligned/doubleadapt_cache_v2/y.npy",
        "hidden":ROOT/"experiments/csi300_aligned/reference/hidden_cache/hidden_test.npy",
        "date_windows":ROOT/"experiments/csi300_aligned/reference/date_windows.csv",
        "old_schedule":HISTORY/"training_schedule.csv"}
    for ckpt,entry in full.MODELS.items():
        inputs[ckpt+"_checkpoint"] = entry["checkpoint"]
        inputs[ckpt+"_reference"] = entry["reference"]
        for variant in ("Static","A"): inputs[ckpt+"_old_"+variant] = HISTORY/"streams"/(ckpt+"_"+variant)/"predictions.csv"
    manifest = {k:{"path":str(p),"sha256":old.sha256(p)} for k,p in inputs.items()}
    old.atomic_json(out/"input_manifest.json",manifest)
    for k in ("runner","full_runner","engine","original_engine"):
        shutil.copy2(inputs[k],out/("executed_"+k+".py"))
    schedule_frame = pd.DataFrame([{"update_date":d,**e} for d,events in schedule.items() for e in events])
    prior_schedule = pd.read_csv(inputs["old_schedule"],dtype={"stock":str})
    pd.testing.assert_frame_equal(schedule_frame,prior_schedule)
    old.atomic_csv(out/"training_schedule.csv",schedule_frame)
    old.atomic_csv(out/"reference_keys.csv",test[KEYS].sort_values(KEYS))
    old.atomic_json(out/"run_config.json",{"dates":dates,"rows":66103,"variants":VARIANTS,"rank":2,"seed":100,
        "training_stocks":stocks,"labels_per_horizon":32,"head_lr":1e-4,"adapter_lr":1e-5,"coeff_lr":1e-4,
        "optimizer":"AdamW","weight_decay":0,"gradient_clip":1,"loss":"SmoothL1 beta1, equal horizon means",
        "dropout":"disabled","updates":216,"supervision":69120,"retrospective":True,"no_test_tuning":True,
        "hidden_sha256":hidden_hash,"half_window":"post_warmup_216 sliced108+108",
        "compute_matched":False,"supervision_steps_dates_matched":True,
        "endpoint_guard":"both return and IC >= A","structure_guard":"PCA Path return and IC >= Random, at least one strict"})
    names,stats,audits,records = [],[],[],{}
    references = {}
    for ckpt,entry in full.MODELS.items():
        assert manifest[ckpt+"_checkpoint"]["sha256"]==entry["checkpoint_sha256"]
        reference = pd.read_csv(entry["reference"],dtype={"stock":str}).sort_values(KEYS).reset_index(drop=True)
        assert reference[KEYS].equals(test[KEYS].sort_values(KEYS).reset_index(drop=True))
        initial = old.make_model(entry["checkpoint"],False)
        with torch.no_grad(): zz = {d:initial.adapter(x) for d,x in hh.items()}
        # Preflight every initial function over the full panel before online updates.
        for variant in VARIANTS:
            model = with_hca(initial,basis if variant=="HCA_PCA" else random_basis()) if variant.startswith("HCA") else copy.deepcopy(initial)
            with torch.no_grad():
                pre = pd.concat([pd.concat([rows[d][KEYS].reset_index(drop=True),pd.DataFrame(model(hh[d]).numpy(),columns=PCOLS)],axis=1) for d in dates]).sort_values(KEYS)
            err = float(np.abs(pre[PCOLS].to_numpy()-reference[PCOLS].to_numpy()).max())
            assert err<=1e-7
            print(json.dumps({"stage":"preflight","checkpoint":ckpt,"variant":variant,"static_error":err}),flush=True)
        references[ckpt] = reference
        canonical_access = None
        for variant in VARIANTS:
            name = ckpt+"_"+variant; names.append(name); stream = out/"streams"/name
            start = with_hca(initial,basis if variant=="HCA_PCA" else random_basis()) if variant.startswith("HCA") else copy.deepcopy(initial)
            model = copy.deepcopy(start); t = time.perf_counter()
            if variant=="Static":
                result = pilot.run_stream("S",model,hh,zz,rows,schedule,pilot.ReleasedLabels(values,releases),stream)
            else:
                result = instrumented_run(model,hh,zz,rows,schedule,pilot.ReleasedLabels(values,releases),stream)
            elapsed = time.perf_counter()-t
            historical = inputs[ckpt+"_old_"+("Static" if variant=="Static" else "A")]
            audit = audit_stream(out,name,result,start,model,dates,reference,historical)
            audits.append(audit)
            if variant!="Static":
                access = result[1].drop(columns="prediction_sha256")
                if canonical_access is None: canonical_access = access
                else: pd.testing.assert_frame_equal(access,canonical_access)
                records[name] = access
            expected_params = 0 if variant=="Static" else (73292 if variant.startswith("HCA") else 72266)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count==expected_params
            stats.append({"model":name,"trainable_parameters":count,"seconds":elapsed,"updates":audit["updates"],
                "supervision":audit["supervision"],**{k+"_delta_l2":v for k,v in audit["parameter_delta_l2"].items()}})
            for suffix,frame in zip(("predictions","access_log","update_log","weight_log"),result):
                if suffix!="predictions" and not frame.empty: old.atomic_csv(stream/(suffix+".csv"),frame)
            # Targets attached after ALL supervised streams have completed below.
            old.atomic_csv(stream/"causal_predictions.csv",result[0])
            print(json.dumps({"stage":"stream_complete",**stats[-1]}),flush=True)
    for access in records.values(): pd.testing.assert_frame_equal(access,next(iter(records.values())))
    for name in names:
        ckpt = name.split("_")[0]
        reference = references[ckpt]
        assert np.allclose(reference[ACOLS],references["Prediction"][ACOLS],atol=1e-7,rtol=0)
        prediction = pd.read_csv(out/"streams"/name/"causal_predictions.csv",dtype={"stock":str},float_precision="round_trip")
        target_ids = prediction[KEYS+["row_id"]].merge(reference[KEYS+ACOLS],on=KEYS,validate="one_to_one")
        np.testing.assert_allclose(np.asarray(values[target_ids.row_id.to_numpy()]),target_ids[ACOLS],atol=1e-7,rtol=0)
        old.atomic_csv(out/"streams"/name/"predictions.csv",full.attach_targets(prediction,reference,name))
    old.atomic_csv(out/"training_summary.csv",pd.DataFrame(stats))
    old.atomic_json(out/"audit.json",{"status":"passed","streams":audits,"matched_events_all_online_streams":True})
    full.evaluate(out,names,dates)
    make_report(out,names,dates)
    for source in manifest.values(): assert old.sha256(Path(source["path"]))==source["sha256"]
    basis_manifest = json.loads((out/"basis_manifest.json").read_text())
    for kind in ("pca","random"):
        assert old.sha256(out/f"basis_{kind}.npy")==basis_manifest["matrices"][kind]["sha256"]
    old.atomic_json(out/"status.json",{"status":"complete","seconds":time.perf_counter()-started,"dates":226,"rows_per_stream":66103,"streams":names})
    files = [p for p in out.rglob("*") if p.is_file() and "checkpoints" not in p.parts]
    old.atomic_json(out/"artifact_manifest.json",{str(p.relative_to(out)):{"bytes":p.stat().st_size,"sha256":old.sha256(p)} for p in files})
    print(f"COMPLETE {out}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--checks-only",action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        status = args.output_dir/"status.json"
        if status.exists() and json.loads(status.read_text()).get("status")=="running":
            old.atomic_json(status,{"status":"failed","error":repr(exc)})
        raise
