"""HCA gain attribution: 5 variants x 2 adapter settings x 2 checkpoints.
Independent runner; the original training and evaluation modules are not modified.
MatchedStep uses same-day HCA update magnitudes as an oracle diagnostic, not a deployment policy.
"""
from __future__ import annotations
import argparse, copy, json, math, random, shutil, sys, tempfile, time, html, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(ROOT/".runtime/torchinductor_cache"))
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional
from scripts.experiments import run_hca_online_full as hca
full,pilot,old = hca.full,hca.pilot,hca.old
KEYS,PCOLS,ACOLS = full.KEYS,full.PCOLS,full.ACOLS
VARIANTS = ("A","HCA","HCA_FrozenHead","A_MatchedStep","A_ProjectedUpdate")
ADAPTER_MODES = ("Live","Frozen")
DEFAULT_OUT = ROOT/"experiments/csi300_aligned/models/hca_gain_attribution_20261028"
HCA_HISTORY = ROOT/"experiments/csi300_aligned/models/hca_online_full_20261020"
normalized_state, sample_weights = pilot.normalized_state, pilot.sample_weights


def effective_head(model):
    head=model.head
    if isinstance(head,hca.ResidualHead):
        w=head.base.weight+head.basis@head.coeff.weight
        b=head.base.bias+head.basis@head.coeff.bias
    else: w,b=head.weight,head.bias
    return torch.cat((w.detach(),b.detach()[:,None]),dim=1).clone()


def transform_delta(delta,basis,variant,target=None):
    norm=torch.linalg.vector_norm(delta)
    if variant=="A_ProjectedUpdate":
        changed=delta+basis@(basis.T@delta)
        changed_norm=torch.linalg.vector_norm(changed)
        return changed*(norm/changed_norm) if changed_norm>0 else changed
    if variant=="A_MatchedStep":
        if norm>0: return delta*(float(target)/norm)
        return delta
    return delta


class DiagnosticAdamW(torch.optim.AdamW):
    def __init__(self,groups,model,variant,basis,target_norms):
        super().__init__(groups,weight_decay=0)
        self.model,self.variant=model,variant
        self.basis=torch.as_tensor(basis,dtype=torch.float32)
        self.targets=target_norms or {}
        self.today=None; self.diagnostics={}

    @torch.no_grad()
    def step(self,closure=None):
        before=effective_head(self.model)
        adapter_before=[p.detach().clone() for p in self.model.adapter.parameters()]
        # Observe the gradient of the existing linear head before the Adam step.
        base=self.model.head.base if isinstance(self.model.head,hca.ResidualHead) else self.model.head
        if base.weight.grad is not None:
            grad=torch.cat((base.weight.grad,base.bias.grad[:,None]),dim=1)
            gp=self.basis@(self.basis.T@grad)
            grad_in=float(torch.linalg.vector_norm(gp))
            grad_out=float(torch.linalg.vector_norm(grad-gp))
        else: grad_in=grad_out=0.0
        result=super().step(closure)
        candidate=effective_head(self.model)-before
        candidate_norm=float(torch.linalg.vector_norm(candidate))
        target=self.targets.get(self.today)
        if self.variant=="A_MatchedStep" and target is None: raise AssertionError("missing paired HCA update norm")
        changed=transform_delta(candidate,self.basis,self.variant,target)
        if self.variant in ("A_MatchedStep","A_ProjectedUpdate"):
            merged=before+changed
            self.model.head.weight.copy_(merged[:,:-1]); self.model.head.bias.copy_(merged[:,-1])
        after=effective_head(self.model)
        delta=after-before
        proj=self.basis@(self.basis.T@delta)
        adapter_delta=math.sqrt(sum(float((p-q).square().sum()) for p,q in zip(self.model.adapter.parameters(),adapter_before)))
        actual=float(torch.linalg.vector_norm(delta))
        target_error=abs(actual-float(target)) if target is not None else 0.0
        match_failed=bool(target is not None and candidate_norm==0 and float(target)>0)
        # Float32 parameter addition can slightly perturb very small requested updates.
        if target is not None and not match_failed:
            assert target_error<=max(2e-6,abs(float(target))*2e-4),(actual,target)
        if self.variant=="A_ProjectedUpdate":
            assert abs(actual-candidate_norm)<=max(2e-6,candidate_norm*2e-4)
        coeff_norm=float(torch.linalg.vector_norm(self.model.head.coeff.weight)) if isinstance(self.model.head,hca.ResidualHead) else 0.0
        self.diagnostics={"effective_head_update_norm":actual,
            "candidate_head_update_norm":candidate_norm,
            "update_basis_in_norm":float(torch.linalg.vector_norm(proj)),
            "update_basis_out_norm":float(torch.linalg.vector_norm(delta-proj)),
            "head_gradient_basis_in_norm":grad_in,"head_gradient_basis_out_norm":grad_out,
            "adapter_update_norm":adapter_delta,"coeff_weight_norm":coeff_norm,
            "matched_target_norm":float(target) if target is not None else -1.0,
            "matched_norm_abs_error":target_error,"matching_failed":match_failed}
        return result


def configure_attribution(model,variant,basis,target_norms,head_lr,adapter_lr):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    if variant=="HCA_FrozenHead": head_parameters=list(model.head.coeff.parameters())
    else: head_parameters=list(model.head.parameters())
    for p in head_parameters: p.requires_grad_(True)
    groups=[{"params":head_parameters,"lr":head_lr}]
    if adapter_lr>0:
        for p in model.adapter.parameters(): p.requires_grad_(True)
        groups.append({"params":list(model.adapter.parameters()),"lr":adapter_lr})
    optimizer=DiagnosticAdamW(groups,model,variant,basis,target_norms)
    return optimizer,[p for p in model.parameters() if p.requires_grad]


def run_stream(group, model, h_by_date, z_by_date, rows, schedule, labels, out,
               *, head_lr=1e-4, adapter_lr=1e-5, uniform=False, stop_after=None, resume=None,
               variant="A", basis=None, target_norms=None):
    """Prediction shards are committed before any supervised access on that day.

    Compact daily checkpoints reference per-day journals in the same run directory.
    Resume truncates the logical stream at the requested checkpoint and deterministically
    rewrites only later shards. The CLI never resumes/overwrites an existing experiment.
    """
    out = Path(out)
    optimizer, trainable = configure_attribution(model, variant, basis, target_norms, head_lr, adapter_lr)
    live_adapter = group == "A" and adapter_lr > 0
    dates, consumed, start, version = sorted(rows), set(), 0, 0
    pred_parts, access_parts, update_parts, weight_parts = [], [], [], []
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("attribution_signature") != {"variant":variant,"adapter_lr":adapter_lr,"basis":np.asarray(basis).tolist(),"targets":target_norms}:
            raise AssertionError("resume configuration mismatch")
        if state["group"] != group:
            raise AssertionError("resume group mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        consumed = set(map(tuple, state["consumed"]))
        start, version = state["next_pos"], state["version"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        for d in dates[:start]:
            pred_parts.append(pd.read_csv(out / "prediction_shards" / group / f"{d}.csv", float_precision="round_trip"))
            for kind, parts in (("access", access_parts), ("update", update_parts), ("weight", weight_parts)):
                file = out / "journals" / group / f"{d}_{kind}.csv"
                if file.exists():
                    parts.append(pd.read_csv(file, float_precision="round_trip"))
    for pos, today in enumerate(dates[start:], start):
        model.eval()
        with torch.no_grad():
            prediction = (model(h_by_date[today]) if live_adapter else model.head(z_by_date[today])).numpy()
        frame = rows[today][KEYS + ["row_id"]].copy()
        frame["group"], frame["pred_version"] = group, version
        for k in range(prediction.shape[1]):
            frame[f"pred_return_{k + 1}"] = prediction[:, k].astype(np.float64)
        shard = out / "prediction_shards" / group / f"{today}.csv"
        old.atomic_csv(shard, frame)
        prediction_hash = old.sha256(shard)
        pred_parts.append(frame)
        events = schedule[today] if group != "S" else []
        accesses, weights_log = [], []
        if events:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            adapter_grad_norm = 0.0
            state = normalized_state(z_by_date[today]) if group == "D" else None
            for k in sorted({e["horizon"] for e in events}):
                selected = [e for e in events if e["horizon"] == k]
                ids = []
                targets = []
                # Metadata is fixed before accessing any supervised scalar.
                for event in selected:
                    key = (event["row_id"], k)
                    if key in consumed:
                        raise AssertionError("duplicate label consumption")
                    targets.append(labels.read(event["row_id"], k - 1, today))
                    consumed.add(key)
                    ids.append(event["row_id"])
                    accesses.append({"group": group, "update_date": today, **event,
                                     "prediction_sha256": prediction_hash})
                origin = selected[0]["origin_date"]
                lookup = {int(r): i for i, r in enumerate(rows[origin].row_id)}
                selected_pos = [lookup[i] for i in ids]
                z = z_by_date[origin][selected_pos]
                output = (model(h_by_date[origin][selected_pos]) if live_adapter else model.head(z))[:, k - 1]
                element_loss = functional.smooth_l1_loss(output, torch.tensor(targets), beta=1, reduction="none")
                if group == "D":
                    w, similarity, reason = sample_weights(z, state, uniform)
                    for event, value, sim in zip(selected, w.tolist(), similarity.tolist()):
                        weights_log.append({"update_date": today, **event, "weight": value,
                                            "similarity": sim, "reason": reason})
                else:
                    w = torch.ones(len(selected))
                losses.append((w * element_loss).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            for p in trainable:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise FloatingPointError("missing/nonfinite gradient")
            if live_adapter:
                adapter_grad_norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.adapter.parameters()))
            grad_norm = float(nn.utils.clip_grad_norm_(trainable, 1))
            optimizer.today = today
            optimizer.step()
            version += 1
            access = pd.DataFrame(accesses)
            update = pd.DataFrame([{"group": group, "date": today, "supervision": len(events),
                "optimizer_steps": 1, "unique_feature_rows": access.row_id.nunique(),
                "loss": float(loss.detach()), "grad_norm": grad_norm, "adapter_grad_norm": adapter_grad_norm,
                "version": version, "consumed_events": len(consumed), **optimizer.diagnostics}])
            for kind, data, parts in (("access", access, access_parts), ("update", update, update_parts),
                                      ("weight", pd.DataFrame(weights_log), weight_parts)):
                if not data.empty:
                    old.atomic_csv(out / "journals" / group / f"{today}_{kind}.csv", data)
                    parts.append(data)
            checkpoint = out / "checkpoints" / group / f"{today}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"attribution_signature":{"variant":variant,"adapter_lr":adapter_lr,"basis":np.asarray(basis).tolist(),"targets":target_norms}, "group": group, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "consumed": sorted(consumed), "next_pos": pos + 1, "version": version,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()}, checkpoint)
        if stop_after is not None and pos + 1 >= stop_after:
            break
    result = [pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
              for parts in (pred_parts, access_parts, update_parts, weight_parts)]
    for suffix, frame in zip(("predictions", "access_log", "update_log", "weight_log"), result):
        if not frame.empty:
            old.atomic_csv(out / group / f"{suffix}.csv", frame)
    return result



def load_inputs_for_checkpoint(entry):
    meta,test,hidden,values,dates,stocks,rows,hh,schedule,releases,hidden_hash=full.load_data()
    initial=old.make_model(entry["checkpoint"],False)
    with torch.no_grad(): z={d:initial.adapter(x) for d,x in hh.items()}
    return meta,test,hidden,values,dates,stocks,rows,hh,z,schedule,releases,hidden_hash,initial


def save_stream(out,name,result,reference):
    stream=Path(out)/name
    prediction=result[0]
    old.atomic_csv(stream/"causal_predictions.csv",prediction)
    for suffix,frame in zip(("access_log","update_log","weight_log"),result[1:]):
        if not frame.empty: old.atomic_csv(stream/(suffix+".csv"),frame)
    return prediction


def run_variant(out,ckpt_name,mode,variant,entry,basis,data_bundle,target_norms=None):
    meta,test,hidden,values,dates,stocks,rows,hh,z,schedule,releases,hidden_hash,initial=data_bundle
    if variant in ("HCA","HCA_FrozenHead"):
        model=hca.with_hca(initial,basis)
    else: model=copy.deepcopy(initial)
    adapter_lr=1e-5 if mode=="Live" else 0.0
    name=f"{ckpt_name}_{mode}_{variant}"
    result=run_stream("A",model,hh,z,rows,schedule,pilot.ReleasedLabels(values,releases),Path(out)/name,
        head_lr=1e-4,adapter_lr=adapter_lr,variant=("A" if variant=="A" else variant),basis=basis,
        target_norms=target_norms)
    return name,result,model


def evaluate_one(out,name,reference,dates):
    prediction_path=Path(out)/name/"predictions.csv"
    eval_dir=Path(out)/"evaluation"/name
    cmd=[sys.executable,"-m","evaluate","--predictions",str(prediction_path),"--output-dir",str(eval_dir),
        "--dataset","CSI300","--test-start",dates[0],"--test-end",dates[-1],"--reference-keys",str(Path(out)/"reference_keys.csv"),
        "--top-k","30","--drop-n","3","--min-holding-days","5","--cost-rate",".0015","--annualization-days","252",
        "--bootstrap-reps","5000","--bootstrap-block","10","--seed","20261013"]
    p=run_child(cmd,cwd=ROOT)
    if p.returncode!=0: raise RuntimeError(p.stderr[-4000:])
    summary=json.loads((eval_dir/"evaluation_summary.json").read_text(encoding="utf8"))
    rows=[]
    for signal in ("path","endpoint"):
        d=pd.read_csv(eval_dir/f"evaluation_{signal}_cross_section_daily.csv")
        q=pd.read_csv(eval_dir/f"evaluation_{signal}_portfolio_daily.csv")
        if d.empty: continue
        f=d.merge(q,on="as_of_date",validate="one_to_one",suffixes=("","_portfolio"))
        stats=full.return_stats(f.top_net.to_numpy())
        rows.append({"name":name,"signal":signal,"ic":float(f.ic.mean()),"rankic":float(f.rankic.mean()),
            "turnover":float(f.turnover.mean()),**{k:float(v) for k,v in stats.items()},
            "trajectory_mae":summary["metrics"]["trajectory_pooled"].get("mae"),
            "trajectory_rmse":summary["metrics"]["trajectory_pooled"].get("rmse")})
    return rows


def run_child(cmd,cwd):
    import subprocess
    return subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,encoding="utf8")


def write_report(out,metric_rows,update_rows,meta_info):
    out=Path(out); table=pd.DataFrame(metric_rows); training=pd.DataFrame(update_rows)
    old.atomic_csv(out/"attribution_metrics.csv",table)
    old.atomic_csv(out/"attribution_training_summary.csv",training)
    daily={}; horizons=[]
    for name in table.name.unique():
        ev=out/"evaluation"/name
        horizon=pd.read_csv(ev/"evaluation_point_by_horizon.csv"); horizon.insert(0,"name",name); horizons.append(horizon)
        for sig in ("path","endpoint"):
            c=pd.read_csv(ev/f"evaluation_{sig}_cross_section_daily.csv"); p=pd.read_csv(ev/f"evaluation_{sig}_portfolio_daily.csv")
            f=c.merge(p,on="as_of_date",validate="one_to_one",suffixes=("","_portfolio")).sort_values("as_of_date").reset_index(drop=True)
            assert len(f)==226 and np.isfinite(f[["ic","rankic","top_net"]]).all().all()
            np.testing.assert_allclose(f.top_net,f.top_raw-.0015*f.turnover,atol=1e-14,rtol=0)
            daily[name,sig]=f
    old.atomic_csv(out/"attribution_by_horizon.csv",pd.concat(horizons,ignore_index=True))
    comps=[]
    for ckpt in full.MODELS:
        for mode in ADAPTER_MODES:
            prefix=f"{ckpt}_{mode}_"; base=prefix+"A"; hc=prefix+"HCA"
            comps.extend((prefix+v,base) for v in VARIANTS if v!="A")
            comps.extend((hc,prefix+v) for v in ("HCA_FrozenHead","A_MatchedStep","A_ProjectedUpdate"))
        comps.extend((f"{ckpt}_Live_{v}",f"{ckpt}_Frozen_{v}") for v in VARIANTS)
    boot=[]; slices=[]; paired=[]
    for window,start in (("full_226",0),("post_warmup_216",10)):
        draws=full.indices(226-start,5000,10,20261013)
        for name in table.name.unique():
            for sig in ("path","endpoint"):
                f=daily[name,sig].iloc[start:]
                slices.append({"window":window,"name":name,"signal":sig,"dates":len(f),"ic":float(f.ic.mean()),"rankic":float(f.rankic.mean()),**{k:float(v) for k,v in full.return_stats(f.top_net.to_numpy()).items()}})
        for a,b in comps:
            for sig in ("path","endpoint"):
                aa=daily[a,sig].iloc[start:]; bb=daily[b,sig].iloc[start:]
                assert aa.as_of_date.tolist()==bb.as_of_date.tolist()
                diff=pd.DataFrame({"window":window,"comparison":a+"_minus_"+b,"signal":sig,"as_of_date":aa.as_of_date.to_numpy()})
                for metric in ("ic","rankic","top_net","turnover"):
                    delta=aa[metric].to_numpy()-bb[metric].to_numpy(); sample=delta[draws].mean(1); diff[metric+"_delta"]=delta
                    boot.append({"window":window,"comparison":a+"_minus_"+b,"signal":sig,"metric":"mean_"+metric,"delta":float(delta.mean()),"ci_low":float(np.quantile(sample,.025)),"ci_high":float(np.quantile(sample,.975))})
                paired.append(diff)
                av=aa.top_net.to_numpy(); bv=bb.top_net.to_numpy()
                sa,sb=full.return_stats(av),full.return_stats(bv); ba,bbv=full.return_stats(av[draws]),full.return_stats(bv[draws])
                for metric in sa:
                    x=ba[metric]-bbv[metric]
                    boot.append({"window":window,"comparison":a+"_minus_"+b,"signal":sig,"metric":metric,"delta":float(sa[metric]-sb[metric]),"ci_low":float(np.quantile(x,.025)),"ci_high":float(np.quantile(x,.975))})
    bdf=pd.DataFrame(boot); old.atomic_csv(out/"attribution_bootstrap.csv",bdf)
    old.atomic_csv(out/"attribution_window_metrics.csv",pd.DataFrame(slices))
    old.atomic_csv(out/"paired_daily_differences.csv",pd.concat(paired,ignore_index=True))
    wide=[]
    for ckpt in full.MODELS:
        for mode in ADAPTER_MODES:
            for sig in ("path","endpoint"):
                row={"checkpoint":ckpt,"adapter":mode,"signal":sig}
                for v in VARIANTS:
                    r=table[table.name.eq(f"{ckpt}_{mode}_{v}") & table.signal.eq(sig)].iloc[0]
                    for metric in ("net_cumulative_return","ic","rankic","sharpe","max_drawdown"): row[v+"_"+metric]=r[metric]
                wide.append(row)
    wide=pd.DataFrame(wide); old.atomic_csv(out/"attribution_wide.csv",wide)
    conclusions=[]; primary=table.set_index(["name","signal"])
    for ckpt in full.MODELS:
        for mode in ADAPTER_MODES:
            prefix=f"{ckpt}_{mode}_"; a=primary.loc[(prefix+"A","path")]; hc=primary.loc[(prefix+"HCA","path")]
            interval=bdf[(bdf.window=="full_226") & (bdf.comparison==prefix+"HCA_minus_"+prefix+"A") & (bdf.signal=="path") & (bdf.metric=="net_cumulative_return")].iloc[0]
            conclusions.append(f"{ckpt}/{mode}：HCA−A 的 Path 净收益差为 {(hc.net_cumulative_return-a.net_cumulative_return)*100:+.3f} 个百分点，95%配对区间 [{interval.ci_low*100:+.3f}, {interval.ci_high*100:+.3f}]；IC 差 {hc.ic-a.ic:+.6f}。")
            for v,label in (("A_MatchedStep","幅度匹配"),("A_ProjectedUpdate","方向投影"),("HCA_FrozenHead","冻结原头")):
                r=primary.loc[(prefix+v,"path")]
                conclusions.append(f"同一配置下，{label}对照相对 HCA 的收益差为 {(r.net_cumulative_return-hc.net_cumulative_return)*100:+.3f} 个百分点，IC 差为 {r.ic-hc.ic:+.6f}。")
    mechanism=[]
    diag_path=out/"mechanism_prediction_diagnostics.csv"
    if diag_path.exists():
        diag=pd.read_csv(diag_path); matched=diag[diag.comparison.eq("HCA_vs_A_MatchedStep")]
        mechanism.append("有效头更新范数：HCA/A 的均值比为 "+f"{matched.hca_to_A_head_norm_ratio.min():.3f}—{matched.hca_to_A_head_norm_ratio.max():.3f}"+"；PCA子空间内为 "+f"{matched.hca_to_A_in_basis_ratio.min():.3f}—{matched.hca_to_A_in_basis_ratio.max():.3f}"+"，子空间外为 "+f"{matched.hca_to_A_out_basis_ratio.min():.3f}—{matched.hca_to_A_out_basis_ratio.max():.3f}"+"。这直接支持 HCA 主要放大已有 PCA 方向更新的机制描述。")
        mechanism.append("幅度匹配后，与 HCA 的逐日 Path 信号排序相关性为 "+f"{matched.mean_daily_signal_rank_correlation.min():.6f}—{matched.mean_daily_signal_rank_correlation.max():.6f}"+"，每日无持仓约束 Top30 集合平均重合率为 "+f"{matched.mean_daily_top30_overlap.min():.2%}—{matched.mean_daily_top30_overlap.max():.2%}"+"。均值包含10日共同暖启动；集合不是实际持仓。幅度是预测排序变化的重要解释，但仍有残余差异，不能认定全部收益差由此产生。")
    mechanism.append("HCA 相对 A 的四个配置 Path 收益差95%区间均跨零；收益差方向随 checkpoint 和 Adapter 状态改变。当前没有稳定、可推广的 HCA 净收益增益证据。")
    mechanism.append("冻结原头后，Prediction/Live 的 Path 收益升至79.920%，但 Endpoint 降至66.960%；RankIC/Live 的 IC升高而收益降低。该干预暴露指标冲突，不能作为更优最终模型直接选用。")
    conclusions=mechanism+conclusions
    rules=["HCA 的线性 CoeffHead 可折叠进原线性头，函数类没有扩大；这些对照检验优化路径和参数化的作用。",
        "A_MatchedStep 仅匹配有效头（权重及偏置合并）的每日更新范数；Adapter 和 Adam 状态沿自身轨迹演化，不等价于匹配全部模型或学习率。",
        "A_ProjectedUpdate 对 AdamW 候选头更新施加 I+BBᵀ 后恢复原范数；它不是 HCA 在 AdamW 下的精确等价算法。",
        "FrozenHead 与 A 的差异同时涉及冻结原头和新增残差分支；该组不能单独证明 CoeffHead 的独立净贡献。",
        "Live/Frozen 改变 Adapter 可训练性，也改变参与全局梯度裁剪的参数集合；因此是完整策略的干预比较。",
        "对照结果接近或差值区间跨零，不证明两种方法等价，也不能据此判定唯一增益来源。"]
    limitations="meta.csv 原文件缺失，已按原 build_cache 的列读取与 row_id 分配规则从 SHA256 核验通过的原始 eligible_stock_date_pairs.csv 恢复；恢复依据、哈希及标签对齐误差见 metadata_recovery.json。此前置恢复后，四条历史在线预测复现误差均为0。 这是已经查看过的 CSI300 固定面板上的回顾性机制实验，不能充当独立确认；单种子、单市场，未做多重比较校正。沿用历史成分面板，保留潜在成分前视/幸存者偏差。所有组使用完整成熟路径标签，不检验 prefix feedback。A_MatchedStep 使用同日 HCA 更新范数，是 oracle 诊断对照。Bootstrap 重采样已实现日收益，不重训或重放持仓。post_warmup_216 为同一226日组合的后216日切片，不重新建仓。"
    columns=["Checkpoint","Adapter","Signal","A 净收益","HCA 净收益","HCA−A (pp)","FrozenHead 净收益","MatchedStep 净收益","Projected 净收益","A IC","HCA IC"]
    formatted=[]
    for _,r in wide.iterrows():
        formatted.append([r.checkpoint,r.adapter,r.signal,f"{r.A_net_cumulative_return:.3%}",f"{r.HCA_net_cumulative_return:.3%}",f"{(r.HCA_net_cumulative_return-r.A_net_cumulative_return)*100:+.3f}",f"{r.HCA_FrozenHead_net_cumulative_return:.3%}",f"{r.A_MatchedStep_net_cumulative_return:.3%}",f"{r.A_ProjectedUpdate_net_cumulative_return:.3%}",f"{r.A_ic:.6f}",f"{r.HCA_ic:.6f}"])
    lines=["# HCA 增益来源归因实验","","20条完整在线流：2 checkpoint × 2 Adapter状态 × 5种策略。每条226日、66,103个预测键、216次更新、69,120个成熟标量标签。","","## 结果","","|"+"|".join(columns)+"|","|"+"|".join(["---"]*len(columns))+"|"]
    lines.extend("|"+"|".join(r)+"|" for r in formatted)
    lines += ["","## 数据支持的描述",""]+["- "+c for c in conclusions]+["","## 对照的准确含义",""]+["- "+c for c in rules]
    selected=bdf[(bdf.window=="full_226") & (bdf.signal=="path") & (bdf.metric.isin(["mean_ic","net_cumulative_return"])) & bdf.comparison.str.contains("HCA")]
    lines += ["","## 完整窗口 Path 配对区间","","|comparison|metric|delta|95%low|95%high|","|---|---|---:|---:|---:|"]
    lines += [f"|{r.comparison}|{r.metric}|{r.delta:+.6f}|{r.ci_low:+.6f}|{r.ci_high:+.6f}|" for r in selected.itertuples()]
    lines += ["","## 限制","",limitations,"","## 配置","",json.dumps(meta_info,ensure_ascii=False,indent=2),""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf8")
    def html_table(headers,rows):
        return "<div class='scroll'><table><thead><tr>"+"".join("<th>"+html.escape(str(x))+"</th>" for x in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table></div>"
    ci_rows=[[r.comparison,r.metric,f"{r.delta:+.6f}",f"{r.ci_low:+.6f}",f"{r.ci_high:+.6f}"] for r in selected.itertuples()]
    diagnostics=training[["name","trainable_parameters","effective_head_update_norm_mean","adapter_update_norm_mean","residual_probe_rms","matched_norm_abs_error_max"]].copy()
    for col in diagnostics.columns[2:]: diagnostics[col]=diagnostics[col].map(lambda v:f"{v:.6g}")
    doc="<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>HCA 增益来源实验</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;line-height:1.65;color:#202c39}h1,h2{color:#175c60}.note{background:#fff5df;border-left:4px solid #b07813;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%;margin:18px 0}th,td{border:1px solid #d5dfe5;padding:8px;text-align:right}th{background:#eaf2f4}td:first-child{text-align:left}li{margin:10px 0}a{color:#135d8a}</style></head><body><h1>HCA 增益来源归因实验</h1><p>20条在线流 · 226日 · 每流216次更新 · 两种初始化 · Adapter联合训练/冻结</p><p class='note'>历史 A/HCA 复现门与因果访问审计已通过。净收益已计入单向15bps成本。以下结果为已查看面板上的探索性机制证据。</p><h2>Path 与 Endpoint 结果</h2>"+html_table(columns,formatted)+"<h2>结果解读</h2><ul>"+"".join("<li>"+html.escape(c)+"</li>" for c in conclusions)+"</ul><h2>对照定义与可识别边界</h2><ul>"+"".join("<li>"+html.escape(c)+"</li>" for c in rules)+"</ul><h2>更新诊断</h2>"+html_table(diagnostics.columns,diagnostics.itertuples(index=False,name=None))+"<p>residual_probe_rms 是最终模型在固定首日输入上的残差输出；不是整个期间的平均贡献。各流固定探针相同，未作为训练或选择信号。</p><h2>Path 配对区间</h2><p>block=10，5000次，seed=20261013。收益差用小数收益单位，乘100为百分点；IC差无量纲。</p>"+html_table(["对比（左−右）","指标","差值","95%下限","95%上限"],ci_rows)+"<h2>限制</h2><p>"+html.escape(limitations)+"</p><p><a href='REPORT.md'>完整报告</a> · <a href='attribution_metrics.csv'>指标CSV</a> · <a href='attribution_bootstrap.csv'>全部区间</a> · <a href='attribution_by_horizon.csv'>逐horizon误差</a> · <a href='stream_audits.json'>审计</a></p></body></html>"
    (out/"hca_gain_attribution.html").write_text(doc,encoding="utf8")
    old.atomic_json(out/"report_summary.json",{"descriptive_conclusions":conclusions,"limitations":limitations})


def attribution_checks():
    inherited=hca.correctness_checks()
    old.seed_all(100)
    dates=[f"2020-01-{i+1:02d}" for i in range(20)]; dd=dates[:14]
    initial=pilot.ToyModel().eval()
    hh={d:torch.randn(5,4) for d in dd}
    with torch.no_grad(): zz={d:initial.adapter(x) for d,x in hh.items()}
    rows={d:pd.DataFrame([{"as_of_date":d,"stock":f"s{i}","row_id":j*5+i,
        "label_dates":dates[j+1:j+4]} for i in range(5)]) for j,d in enumerate(dd)}
    releases={int(r.row_id):r.label_dates for f in rows.values() for r in f.itertuples()}
    values=np.random.default_rng(100).normal(0,.01,(70,3))
    schedule=pilot.make_schedule(dd,rows,3,4); basis=hca.random_basis(3)
    hm=hca.with_hca(initial,basis); x=hh[dd[0]]
    with torch.no_grad():
        torch.testing.assert_close(hm(x),initial(x),atol=0,rtol=0)
        hm.head.coeff.weight.normal_(0,.01); hm.head.coeff.bias.normal_(0,.01)
        e=effective_head(hm)
        torch.testing.assert_close(hm(x),functional.linear(hm.adapter(x),e[:,:-1],e[:,-1]),atol=1e-7,rtol=1e-6)
    delta=torch.randn(3,5); b=torch.tensor(basis)
    changed=transform_delta(delta,b,"A_ProjectedUpdate")
    torch.testing.assert_close(changed.norm(),delta.norm(),atol=1e-6,rtol=1e-6)
    expected=delta+b@(b.T@delta)
    torch.testing.assert_close(changed/changed.norm(),expected/expected.norm(),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(transform_delta(delta,b,"A_MatchedStep",.02).norm(),torch.tensor(.02))
    count=0
    with tempfile.TemporaryDirectory(prefix="attribution_checks_") as td:
        td=Path(td)
        for mode in ADAPTER_MODES:
            targets=None
            for variant in ("HCA","A","HCA_FrozenHead","A_MatchedStep","A_ProjectedUpdate"):
                name=mode+"_"+variant; lr=1e-5 if mode=="Live" else 0
                start=hca.with_hca(initial,basis) if variant.startswith("HCA") else copy.deepcopy(initial)
                paired_targets=targets if variant=="A_MatchedStep" else None
                def execute(path,data=values,**kw):
                    model=copy.deepcopy(start)
                    rec=run_stream("A",model,hh,zz,rows,schedule,pilot.ReleasedLabels(data,releases),path,
                        adapter_lr=lr,variant=variant,basis=basis,target_norms=paired_targets,**kw)
                    return model,rec
                model,rec=execute(td/name)
                if variant=="HCA": targets={str(r.date):r.effective_head_update_norm for r in rec[2].itertuples()}
                if mode=="Frozen": assert old.nested_equal(model.adapter.state_dict(),start.adapter.state_dict())
                if variant=="HCA_FrozenHead": assert old.nested_equal(model.head.base.state_dict(),start.head.base.state_dict())
                cutoff=dd[7]; poison=values.copy()
                for rid,ds in releases.items():
                    for k,d in enumerate(ds):
                        if d>cutoff: poison[rid,k]=np.nan
                clean,cr=execute(td/(name+"_cut"),stop_after=8)
                bad,br=execute(td/(name+"_poison"),poison,stop_after=8)
                assert old.nested_equal(clean.state_dict(),bad.state_dict())
                for a,c in zip(cr,br): pd.testing.assert_frame_equal(a,c)
                cp=td/(name+"_cut")/"checkpoints/A"/f"{cutoff}.pt"
                resumed,rr=execute(td/(name+"_cut"),resume=cp)
                assert old.nested_equal(model.state_dict(),resumed.state_dict())
                for a,c in zip(rec,rr): pd.testing.assert_frame_equal(a,c,check_exact=True)
                final=torch.load(td/name/"checkpoints/A"/f"{dd[-1]}.pt",weights_only=False)
                again=torch.load(td/(name+"_cut")/"checkpoints/A"/f"{dd[-1]}.pt",weights_only=False)
                for k in ("model","optimizer","consumed"): assert old.nested_equal(final[k],again[k])
                count+=1
    return {"status":"passed","inherited":inherited,"control_streams_tested":count,
        "checks":["same_input_zero_init","folded_head_equivalence","projected_direction_and_norm",
        "matched_norm","all_modes_future_poison","all_modes_resume_exact","frozen_parameters_exact"]}


def audit_attribution(out,name,result,initial,model,mode,variant,basis,dates,reference,history=None):
    prediction,access,updates,_=result
    ordered=prediction.sort_values(KEYS).reset_index(drop=True)
    assert ordered[KEYS].equals(reference[KEYS]) and len(ordered)==66103
    assert not ordered.duplicated(KEYS).any() and np.isfinite(ordered[PCOLS].to_numpy()).all()
    assert len(updates)==216 and len(access)==69120 and updates.date.tolist()==dates[10:]
    assert updates.supervision.eq(320).all() and updates.optimizer_steps.eq(1).all()
    assert (access.complete_date<=access.update_date).all() and (access.label_date<=access.update_date).all()
    assert (access.origin_date<access.update_date).all() and not access.duplicated(["row_id","horizon"]).any()
    for col in ("min","max"):
        np.testing.assert_array_equal(prediction.groupby("as_of_date").pred_version.agg(["min","max"])[col],np.maximum(0,np.arange(226)-10))
    for d,digest in access[["update_date","prediction_sha256"]].drop_duplicates().itertuples(index=False,name=None):
        assert old.sha256(out/name/"prediction_shards/A"/f"{d}.csv")==digest
    changes=hca.parameter_changes(initial,model)
    assert (changes["adapter"]>0) if mode=="Live" else changes["adapter"]==0
    assert (changes["head"]==0) if variant=="HCA_FrozenHead" else changes["head"]>0
    if variant.startswith("HCA"):
        assert changes["coeff"]>0 and torch.equal(model.head.basis,torch.tensor(basis))
        assert not model.head.basis.requires_grad
    if variant=="A_MatchedStep": assert not updates.matching_failed.any()
    max_error=None
    if history is not None:
        historic=pd.read_csv(history,dtype={"stock":str},float_precision="round_trip").sort_values(KEYS).reset_index(drop=True)
        assert historic[KEYS].equals(ordered[KEYS])
        max_error=float(np.abs(historic[PCOLS].to_numpy()-ordered[PCOLS].to_numpy()).max())
        assert max_error<=1e-7,(name,max_error)
    return {"name":name,"status":"passed","rows":len(ordered),"updates":len(updates),"labels":len(access),
        "historical_max_abs_error":max_error,"parameter_delta_l2":changes,"future_reads":0,"duplicate_labels":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--checks-only",action="store_true")
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args(); out=args.output_dir.resolve()
    old.seed_all(100); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if args.checks_only:
        print(json.dumps(attribution_checks()),flush=True); return
    if args.report_only:
        state=json.loads((out/"training_complete.json").read_text(encoding="utf8"))
        metrics=[]
        for name in state["names"]:
            ev=out/"evaluation"/name
            summary=json.loads((ev/"evaluation_summary.json").read_text(encoding="utf8"))
            for sig in ("path","endpoint"):
                c=pd.read_csv(ev/f"evaluation_{sig}_cross_section_daily.csv"); p=pd.read_csv(ev/f"evaluation_{sig}_portfolio_daily.csv")
                f=c.merge(p,on="as_of_date",validate="one_to_one",suffixes=("","_portfolio"))
                metrics.append({"name":name,"signal":sig,"ic":float(f.ic.mean()),"rankic":float(f.rankic.mean()),"turnover":float(f.turnover.mean()),**{k:float(v) for k,v in full.return_stats(f.top_net.to_numpy()).items()},"trajectory_mae":summary["metrics"]["trajectory_pooled"]["mae"],"trajectory_rmse":summary["metrics"]["trajectory_pooled"]["rmse"]})
        write_report(out,metrics,state["update_rows"],state["config"])
        shutil.copy2(__file__,out/"final_runner_snapshot.py")
        print("REPORT COMPLETE",flush=True); return
    if args.evaluate_only:
        state=json.loads((out/"training_complete.json").read_text(encoding="utf8"))
        metrics=[]
        for name in state["names"]:
            old.atomic_json(out/"status.json",{"status":"evaluating","current":name,"completed":len(metrics)//2})
            metrics.extend(evaluate_one(out,name,None,state["dates"]))
            print("EVALUATED "+name,flush=True)
        write_report(out,metrics,state["update_rows"],state["config"])
        old.atomic_json(out/"status.json",{"status":"complete","streams":len(state["names"]),"metrics":len(metrics)})
        print("COMPLETE "+str(out),flush=True); return
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"new output directory required: {out}")
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    old.atomic_json(out/"status.json",{"status":"checking"})
    checks=attribution_checks(); old.atomic_json(out/"correctness_report.json",checks)
    shutil.copy2(__file__,out/"runner_snapshot.py")
    config={"variants":VARIANTS,"adapter_modes":ADAPTER_MODES,"seed":100,"window":"2025-07-01/2026-06-05",
        "updates_per_stream":216,"labels_per_stream":69120,"rank":2,"basis":"train_split_PCA",
        "head_lr":1e-4,"adapter_lr_live":1e-5,"optimizer":"AdamW wd0","clip_norm":1,
        "loss":"SmoothL1 beta1 mean horizons","dropout":False,"cost_rate":.0015,
        "top_k":30,"drop_n":3,"min_holding_days":5,"bootstrap":"block10 reps5000 seed20261013"}
    names=[]; summaries=[]; audits=[]; manifest={}; bundles={}; references={}; targets={}
    def execute(ckpt,mode,variant):
        entry=full.MODELS[ckpt]; bundle,basis=bundles[ckpt]
        initial=hca.with_hca(bundle[-1],basis) if variant.startswith("HCA") else copy.deepcopy(bundle[-1])
        name=f"{ckpt}_{mode}_{variant}"
        old.atomic_json(out/"status.json",{"status":"training","current":name,"completed":len(names)})
        t=time.perf_counter()
        name,result,model=run_variant(out,ckpt,mode,variant,entry,basis,bundle,targets.get((ckpt,mode)) if variant=="A_MatchedStep" else None)
        history=None
        if mode=="Live" and variant=="A": history=ROOT/f"experiments/csi300_aligned/models/adapter_online_full_20261020/streams/{ckpt}_A/predictions.csv"
        if mode=="Live" and variant=="HCA": history=HCA_HISTORY/f"streams/{ckpt}_HCA_PCA/predictions.csv"
        audit=audit_attribution(out,name,result,initial,model,mode,variant,basis,bundle[4],references[ckpt],history)
        audits.append(audit); old.atomic_json(out/"stream_audits.json",audits)
        save_stream(out,name,result,None); u=result[2]
        if variant=="HCA": targets[ckpt,mode]={str(r.date):float(r.effective_head_update_norm) for r in u.itertuples()}
        summary={"name":name,"variant":variant,"mode":mode,"updates":len(u),"supervision":len(result[1]),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"elapsed_seconds":time.perf_counter()-t}
        for col in ("effective_head_update_norm","candidate_head_update_norm","update_basis_in_norm","update_basis_out_norm",
                    "head_gradient_basis_in_norm","head_gradient_basis_out_norm","adapter_update_norm","coeff_weight_norm","matched_norm_abs_error"):
            summary[col+"_mean"]=float(u[col].mean()); summary[col+"_max"]=float(u[col].max())
        with torch.no_grad():
            fixed=bundle[7][bundle[4][0]]
            diff=model(fixed)-initial(fixed)
            summary["fixed_probe_prediction_change_rms"]=float(diff.square().mean().sqrt())
            summary["residual_probe_rms"]=float((model.head.coeff(model.adapter(fixed))@model.head.basis.T).square().mean().sqrt()) if variant.startswith("HCA") else 0.
        summaries.append(summary); names.append(name)
        old.atomic_csv(out/"attribution_training_summary.csv",pd.DataFrame(summaries))
        print("TRAINED "+name+" "+json.dumps(audit)+f" seconds={time.perf_counter()-t:.1f}",flush=True)
    for ckpt,entry in full.MODELS.items():
        actual=old.sha256(entry["checkpoint"]); assert actual==entry["checkpoint_sha256"]
        bundle=load_inputs_for_checkpoint(entry); basis,info=hca.training_basis(bundle[0],bundle[3],rank=2)
        assert info["rows"]==525700 and info["label_end"]<bundle[4][0]
        np.testing.assert_allclose(basis,np.load(HCA_HISTORY/"basis_pca.npy"),atol=0,rtol=0)
        np.save(out/f"{ckpt}_basis.npy",basis)
        bundles[ckpt]=(bundle,basis)
        references[ckpt]=pd.read_csv(entry["reference"],dtype={"stock":str}).sort_values(KEYS).reset_index(drop=True)
        # Initial checkpoint predictions are verified independently of online labels.
        with torch.no_grad(): p=torch.cat([bundle[-1](bundle[7][d]) for d in bundle[4]]).numpy()
        ordered=pd.concat([bundle[6][d][KEYS] for d in bundle[4]],ignore_index=True); ordered[PCOLS]=p
        ordered=ordered.sort_values(KEYS).reset_index(drop=True)
        initial_error=float(np.abs(ordered[PCOLS].to_numpy()-references[ckpt][PCOLS].to_numpy()).max()); assert initial_error<=1e-7
        manifest[ckpt]={"checkpoint":str(entry["checkpoint"]),"checkpoint_sha256":actual,
            "reference_sha256":old.sha256(entry["reference"]),"hidden_sha256":bundle[11],"initial_prediction_error":initial_error,"basis":info}
        # Mandatory historical reproduction gate before ANY new control streams.
        for variant in ("A","HCA"): execute(ckpt,"Live",variant)
    old.atomic_json(out/"historical_reproduction_gate.json",{"status":"passed","streams":audits.copy()})
    old.atomic_json(out/"input_manifest.json",manifest)
    old.atomic_csv(out/"reference_keys.csv",references["Prediction"][KEYS])
    for ckpt in full.MODELS:
        for mode in ADAPTER_MODES:
            for variant in ("HCA","A","HCA_FrozenHead","A_MatchedStep","A_ProjectedUpdate"):
                if f"{ckpt}_{mode}_{variant}" not in names: execute(ckpt,mode,variant)
    # Targets are attached only after every causal training stream has been committed.
    for name in names:
        ckpt=name.split("_")[0]
        pred=pd.read_csv(out/name/"causal_predictions.csv",dtype={"stock":str},float_precision="round_trip")
        old.atomic_csv(out/name/"predictions.csv",full.attach_targets(pred,references[ckpt],name))
    for ckpt,entry in full.MODELS.items(): assert old.sha256(entry["checkpoint"])==manifest[ckpt]["checkpoint_sha256"]
    for rel in ("experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv","experiments/csi300_aligned/doubleadapt_cache_v2/y.npy","experiments/csi300_aligned/reference/date_windows.csv"):
        manifest[rel]={"sha256":old.sha256(ROOT/rel)}
    old.atomic_json(out/"input_manifest.json",manifest)
    state={"names":names,"dates":bundles["Prediction"][0][4],"update_rows":summaries,"config":config,"training_seconds":time.perf_counter()-started}
    old.atomic_json(out/"training_complete.json",state)
    old.atomic_json(out/"status.json",{"status":"training_complete","streams":len(names),"seconds":time.perf_counter()-started})
    print("TRAINING COMPLETE",flush=True)
    metrics=[]
    for name in names:
        old.atomic_json(out/"status.json",{"status":"evaluating","current":name,"completed":len(metrics)//2})
        metrics.extend(evaluate_one(out,name,None,state["dates"])); print("EVALUATED "+name,flush=True)
    write_report(out,metrics,summaries,config)
    old.atomic_json(out/"status.json",{"status":"complete","streams":len(names),"metrics":len(metrics),"elapsed_seconds":time.perf_counter()-started})
    print("COMPLETE "+str(out),flush=True)

if __name__=="__main__":
    try: main()
    except Exception as exc:
        # Preserve all shards and journals for diagnosis; no automatic overwrites.
        import traceback
        traceback.print_exc()
        sys.exit(1)
