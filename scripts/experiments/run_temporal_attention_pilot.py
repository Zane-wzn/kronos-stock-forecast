"""Bounded CPU temporal-pooling pilot with chronological offline and online evaluation.

All architectures use the same frozen, re-extracted Kronos sequences and original
Prediction Adapter checkpoint. Test labels are released only after complete
maturity in the online loop. Test forecasts never select a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".runtime/torchinductor_cache"))
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.data.build_continual_csi300_hidden_cache import normalise_ohlcv, stamps
from scripts.experiments.run_delayed_feedback_pilot import Model, sha256, stable_stock_key
from scripts.experiments.run_online_module_pilot import make_schedule, ReleasedLabels
from scripts.experiments.run_temporal_attention_smoke import read_npy_rows
from src.kronos_continual.attention_adapter import TemporalAttentionModel
from evaluate.metrics import EvaluationConfig, evaluate_frame, clean_json

ARCHITECTURES = ("adapter", "temporal_mean", "temporal_attention")
PCOLS = [f"pred_return_{k}" for k in range(1, 11)]
ACOLS = [f"actual_return_{k}" for k in range(1, 11)]
KEYS = ["as_of_date", "stock"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def select_panel(meta, stocks=60, train_dates=80, val_dates=40, test_dates=80):
    # Choose cohort from TRAIN availability only; hash is fixed before label access.
    cohort = sorted(set(meta.loc[meta.split.eq("train"), "stock"]),
                    key=lambda s: hashlib.sha256(("temporal-pilot-v1|" + s).encode()).hexdigest())[:stocks]
    selected = []
    for split, count in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
        part = meta[meta.split.eq(split) & meta.stock.isin(cohort)].copy()
        dates = sorted(part.as_of_date.unique())
        if len(dates) < count:
            raise ValueError(f"Not enough {split} dates")
        # Historical train dates spread over the last four years; validation
        # dates spread over the original validation window; test is contiguous.
        if split == "train":
            cutoff = str((pd.Timestamp(dates[-1]) - pd.DateOffset(years=4)).date())
            available = [d for d in dates if d >= cutoff]
            dates = [available[i] for i in np.linspace(0, len(available)-1, count, dtype=int)]
        elif split == "val":
            dates = [dates[i] for i in np.linspace(0, len(dates)-1, count, dtype=int)]
        else:
            dates = dates[:count]
        selected.append(part[part.as_of_date.isin(dates)].sort_values(KEYS))
    rows = pd.concat(selected, ignore_index=True)
    if rows.duplicated(KEYS).any():
        raise AssertionError("duplicate panel keys")
    train, val, test = [rows[rows.split.eq(s)] for s in ("train", "val", "test")]
    if train.label_end_date.max() >= val.as_of_date.min() or val.label_end_date.max() >= test.as_of_date.min():
        raise AssertionError("train/val/test label boundary overlap")
    return rows, cohort


def tensor_batch(h, ids):
    return torch.from_numpy(np.array(h[np.asarray(ids, dtype=int)], dtype=np.float32))


def forward_batch(model, arch, x):
    return model(x[:, -1]) if arch == "adapter" else model(x)


def predict(model, arch, h, ids, batch_size=64):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chunks.append(forward_batch(model, arch, tensor_batch(h, ids[start:start+batch_size])).numpy())
    return np.concatenate(chunks)


def build_sequences(rows, out, args, deadline):
    """Cache one full sequence per selected row, never reconstruct from last hidden."""
    source = ROOT / "third_party/Kronos_official_src/Kronos-master"
    sys.path.insert(0, str(source))
    from model import Kronos, KronosTokenizer
    hub = ROOT / ".runtime/hf/hub"
    model_dir = sorted((hub / "models--NeoQuasar--Kronos-small/snapshots").iterdir())[0]
    tok_dir = sorted((hub / "models--NeoQuasar--Kronos-Tokenizer-base/snapshots").iterdir())[0]
    backbone = Kronos.from_pretrained(str(model_dir)).eval().requires_grad_(False)
    tok = KronosTokenizer.from_pretrained(str(tok_dir)).eval().requires_grad_(False)
    def tensor_hash(module):
        digest = hashlib.sha256()
        for name, value in module.state_dict().items():
            digest.update(name.encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen = {"backbone": tensor_hash(backbone), "tokenizer": tensor_hash(tok)}
    # These windows are market-calendar aligned, same input dates for every stock.
    windows = pd.read_csv(ROOT / "experiments/csi300_aligned/reference/date_windows.csv", dtype=str,
                          usecols=["as_of_date", "input_dates", "label_dates"]).drop_duplicates("as_of_date").set_index("as_of_date")
    frames, raw_hashes = {}, {}
    for stock in rows.stock.unique():
        path = ROOT / "data/csi300_daily/csi300_daily" / (stock.replace(".", "_") + ".csv")
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        if frame.date.duplicated().any(): raise AssertionError("duplicate raw date")
        if "amount" not in frame:
            frame["amount"] = frame[["open", "high", "low", "close"]].mean(axis=1) * frame.volume
        frames[stock] = frame.set_index("date")
        raw_hashes[stock] = sha256(path)
    hpath = out / "cache/hidden_seq.npy"
    hpath.parent.mkdir()
    h = np.lib.format.open_memmap(hpath, mode="w+", dtype="float32", shape=(len(rows),90,512))
    inputs_hash = hashlib.sha256()
    started = time.perf_counter(); done = 0
    with torch.no_grad():
        for (split, date), group in rows.groupby(["split", "as_of_date"], sort=False):
            inp = pd.to_datetime(windows.loc[date, "input_dates"].split("|"))
            if len(inp) != 90 or str(inp[-1].date()) != date: raise AssertionError("bad input boundary")
            for begin in range(0, len(group), 16):
                part = group.iloc[begin:begin+16]
                x = np.stack([normalise_ohlcv(frames[s].loc[inp]) for s in part.stock])
                if not np.isfinite(x).all(): raise AssertionError("nonfinite raw input")
                ts = np.repeat(stamps(inp)[None], len(part), axis=0)
                inputs_hash.update(x.tobytes()); inputs_hash.update(ts.tobytes())
                token = tok.encode(torch.from_numpy(x), half=True)
                _, context = backbone.decode_s1(token[0], token[1], torch.from_numpy(ts))
                arr = context.numpy()
                if not np.isfinite(arr).all(): raise AssertionError("nonfinite context")
                h[part.index.to_numpy()] = arr
            done += len(group)
            if done % 1000 < len(group) or done == len(rows):
                h.flush()
                progress = {"stage":"feature_extraction", "done":done, "total":len(rows),
                            "seconds":time.perf_counter()-started, "split":split,"date":date}
                write_json(out / "progress.json", progress); print(json.dumps(progress), flush=True)
            if time.perf_counter() > deadline - 600: raise TimeoutError("feature extraction exceeds time budget")
    h.flush()
    assert frozen == {"backbone":tensor_hash(backbone), "tokenizer":tensor_hash(tok)}
    del backbone, tok, frames
    info = {"shape":list(h.shape),"dtype":str(h.dtype),"seconds":time.perf_counter()-started,
            "frozen_tensor_hash":frozen,"model_snapshot":model_dir.name,"tokenizer_snapshot":tok_dir.name,
            "input_sha256":inputs_hash.hexdigest(),"raw_sha256":raw_hashes,"sequence_sha256":sha256(hpath)}
    write_json(out / "sequence_cache_manifest.json", info)
    return h, info


def load_model(arch, state, seed):
    torch.manual_seed(seed)
    if arch == "adapter":
        model = Model()
        model.load_state_dict(state)
    else:
        model = TemporalAttentionModel(mode=arch)
        model.adapter.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith("adapter.")})
        model.head.load_state_dict({k[5:]:v for k,v in state.items() if k.startswith("head.")})
    return model.eval()


def optimizer_for(model, arch, offline=False):
    groups=[{"params":model.head.parameters(),"lr":1e-4},
            {"params":model.adapter.parameters(),"lr":1e-5}]
    if arch != "adapter":
        groups.append({"params":[p for p in model.pool.parameters() if p.requires_grad],"lr":1e-5})
    return torch.optim.AdamW(groups,weight_decay=0)


def train_offline(models, rows, h, labels, out, args, deadline):
    train_ids = np.flatnonzero(rows.split.eq("train")); val_ids = np.flatnonzero(rows.split.eq("val"))
    # Time is checked only after a complete paired round, never stopping one
    # architecture earlier and treating the remainder as a matched experiment.
    history=[]; best={}; opts={}; global_step=0
    for key, model in models.items():
        arch, seed = key
        opts[key]=optimizer_for(model,arch,True)
        p=predict(model,arch,h,val_ids); err=p-labels[val_ids]
        best[key]=(float(np.square(err).mean()),0,copy.deepcopy(model.state_dict()))
        history.append({"architecture":arch,"seed":seed,"epoch":0,"step":0,"val_mse":best[key][0],"val_mae":float(abs(err).mean()),"seconds":0})
    finished=0; grad={}; max_epochs=args.epochs
    for epoch in range(1,max_epochs+1):
        # Same sample order and batch membership for all three architectures per seed.
        orders={seed:np.random.default_rng(seed+epoch).permutation(train_ids) for seed in args.seeds}
        for (arch,seed),model in models.items():
            opt=opts[arch,seed]; started=time.perf_counter(); train_loss=[]
            order=orders[seed]
            for begin in range(0,len(order),args.batch_size):
                ix=order[begin:begin+args.batch_size]
                opt.zero_grad(set_to_none=True)
                p=forward_batch(model,arch,tensor_batch(h,ix))
                loss=nn.functional.smooth_l1_loss(p,torch.from_numpy(labels[ix]))
                loss.backward()
                for name,value in model.named_parameters():
                    if value.requires_grad and (value.grad is None or not torch.isfinite(value.grad).all()):
                        raise AssertionError(f"missing/nonfinite gradient {arch}/{name}")
                if epoch==1 and begin==args.batch_size and arch=="temporal_attention":
                    grad[str(seed)]={name:float(value.grad.norm()) for name,value in model.pool.named_parameters()}
                nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); train_loss.append(float(loss.detach()))
            p=predict(model,arch,h,val_ids); err=p-labels[val_ids]
            mse=float(np.square(err).mean())
            if mse<best[arch,seed][0]: best[arch,seed]=(mse,epoch,copy.deepcopy(model.state_dict()))
            rec={"architecture":arch,"seed":seed,"epoch":epoch,"step":epoch*math.ceil(len(train_ids)/args.batch_size),
                 "train_loss":float(np.mean(train_loss)),"val_mse":mse,"val_mae":float(abs(err).mean()),"seconds":time.perf_counter()-started}
            history.append(rec); print(json.dumps({"stage":"offline",**rec}),flush=True)
        finished=epoch
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        write_json(out/"progress.json",{"stage":"offline","epochs_complete":epoch,"max_epochs":max_epochs})
        if time.perf_counter() > deadline - 900: break
    for (arch,seed),model in models.items():
        model.load_state_dict(best[arch,seed][2])
        torch.save({"model":model.state_dict(),"selected_epoch":best[arch,seed][1],"validation_mse":best[arch,seed][0]},out/f"checkpoints/{arch}_{seed}_offline.pt")
    write_json(out/"gradient_diagnostics.json",grad)
    write_json(out/"selection.json",{f"{a}_{s}":{"epoch":v[1],"validation_mse":v[0]} for (a,s),v in best.items()})
    return best,finished


def prediction_frame(rows, ix, p, labels, arch,seed,phase,version=0):
    frame=rows.iloc[ix][KEYS+["row_id","label_start_date","label_end_date"]].copy()
    frame[PCOLS]=p
    if labels is not None: frame[ACOLS]=labels[ix]
    frame["pred_signal"]=p.mean(1); frame["pred_endpoint_return"]=p[:,-1]
    if labels is not None:
        frame["actual_signal"]=labels[ix].mean(1);frame["actual_endpoint_return"]=labels[ix,-1]
    frame["architecture"]=arch;frame["method"]=arch+"_"+phase;frame["seed"]=seed;frame["version"]=version
    return frame.reset_index(drop=True)


def online_predictions(models,rows,h,labels,out,args):
    test=rows[rows.split.eq("test")]; dates=sorted(test.as_of_date.unique())
    windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str,
                        usecols=['as_of_date','label_dates']).drop_duplicates('as_of_date').set_index('as_of_date')
    table={}
    for d in dates:
        f=test[test.as_of_date.eq(d)].copy()
        f['row_id']=f.index.to_numpy()  # local cache/label address, global identity also kept in selected_samples
        f['label_dates']=[windows.loc[d,'label_dates'].split('|')]*len(f)
        table[d]=f.reset_index(drop=True)
    schedule=make_schedule(dates,table,max_rows=32)
    releases={int(r.row_id):r.label_dates for f in table.values() for r in f.itertuples()}
    guarded=ReleasedLabels(labels,releases)
    frames={}; accesses=[]; logs=[]; attn=[]; timings=[]
    expected_events=None
    for (arch,seed),model in models.items():
        opt=optimizer_for(model,arch); version=0; consumed=set(); parts=[]; model_events=[]
        for pos,d in enumerate(dates):
            ix=test[test.as_of_date.eq(d)].index.to_numpy()
            start=time.perf_counter(); pred=predict(model,arch,h,ix); predict_sec=time.perf_counter()-start
            issued=prediction_frame(rows,ix,pred,None,arch,seed,'online',version)
            shard=out/f"issued/{arch}_{seed}/{d}.csv";shard.parent.mkdir(parents=True,exist_ok=True)
            issued.to_csv(shard,index=False);digest=sha256(shard)
            events=schedule[d];update_sec=0
            if events:
                start=time.perf_counter();opt.zero_grad(set_to_none=True);losses=[]
                for k in sorted({e['horizon'] for e in events}):
                    selected=[e for e in events if e['horizon']==k]; ids=[];ys=[]
                    for event in selected:
                        key=(event['row_id'],k)
                        if key in consumed: raise AssertionError('duplicate consumption')
                        ys.append(guarded.read(event['row_id'],k-1,d));ids.append(event['row_id']);consumed.add(key)
                        event_key=(d,event['row_id'],k);model_events.append(event_key)
                        accesses.append({'architecture':arch,'seed':seed,'update_date':d,**event,'prediction_sha256':digest})
                    pp=forward_batch(model,arch,tensor_batch(h,ids))[:,k-1]
                    losses.append(nn.functional.smooth_l1_loss(pp,torch.tensor(ys,dtype=torch.float32)))
                loss=torch.stack(losses).mean();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();version+=1
                update_sec=time.perf_counter()-start
                logs.append({'architecture':arch,'seed':seed,'date':d,'version_after':version,'supervision':len(events),'loss':float(loss.detach())})
            # Attach labels only after the complete prediction stream, below.
            parts.append(issued)
            timings.append({'architecture':arch,'seed':seed,'date':d,'predict_seconds':predict_sec,'update_seconds':update_sec})
            if arch=='temporal_attention' and pos in (0,len(dates)-1):
                with torch.no_grad(): _,weights=model.pool(tensor_batch(h,ix),return_weights=True)
                attn.append({'seed':seed,'date':d,'when':'after_update','mean_weights':weights.mean(0).tolist(),
                             'mean_entropy':float(-(weights*weights.clamp_min(1e-12).log()).sum(1).mean())})
        if expected_events is None: expected_events=model_events
        elif expected_events!=model_events: raise AssertionError('unmatched online feedback')
        full=pd.concat(parts,ignore_index=True)
        # original global row_id is not a local index; look up by canonical keys.
        lookup=rows.reset_index().set_index(KEYS)['index']
        loc=np.array([lookup.loc[(r.as_of_date,r.stock)] for r in full.itertuples()])
        full[ACOLS]=labels[loc];full['actual_signal']=labels[loc].mean(1);full['actual_endpoint_return']=labels[loc,-1]
        frames[arch,seed,'online']=full
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'consumed':sorted(consumed),'version':version},out/f'checkpoints/{arch}_{seed}_online.pt')
        print(json.dumps({'stage':'online','architecture':arch,'seed':seed,'updates':version,'events':len(consumed)}),flush=True)
    pd.DataFrame(accesses).to_csv(out/'label_access_log.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'online_update_log.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'online_timings.csv',index=False)
    write_json(out/'attention_weights.json',attn)
    return frames


def summarize(frames,out,args):
    summaries=[];daily={};horizons=[]
    for (arch,seed,phase),frame in frames.items():
        name=f'{arch}_{seed}_{phase}'; dest=out/'evaluation'/name; dest.mkdir(parents=True)
        frame.to_csv(dest/'predictions.csv',index=False)
        cfg=EvaluationConfig(test_start=frame.as_of_date.min(),test_end=frame.as_of_date.max(),
                             top_k=6,drop_n=1,min_holding_days=5,cost_rate=.0015,bootstrap_reps=1000,bootstrap_block=10,seed=20261104)
        result=evaluate_frame(frame,cfg)
        write_json(dest/'evaluation_summary.json',result)
        p=frame[PCOLS].to_numpy();y=frame[ACOLS].to_numpy();e=p-y
        point_daily=[]
        for date,g in frame.groupby('as_of_date'):
            err=g[PCOLS].to_numpy()-g[ACOLS].to_numpy()
            point_daily.append({'as_of_date':date,'mae':float(abs(err).mean()),'mse':float((err**2).mean()),
                                'direction':float(((g[PCOLS].to_numpy()>0)==(g[ACOLS].to_numpy()>0)).mean())})
        point_daily=pd.DataFrame(point_daily)
        for mode in ('path','endpoint'):
            m=result['metrics'][mode]; cross=pd.DataFrame(result['daily'][mode]['cross_section']); port=pd.DataFrame(result['daily'][mode]['portfolio'])
            dd=point_daily.merge(cross,on='as_of_date',validate='one_to_one').merge(port,on='as_of_date',validate='one_to_one',suffixes=('','_portfolio'))
            dd.to_csv(dest/f'{mode}_daily.csv',index=False);daily[arch,seed,phase,mode]=dd
            summaries.append({'architecture':arch,'seed':seed,'phase':phase,'signal':mode,'rows':len(frame),'dates':frame.as_of_date.nunique(),
                 'trajectory_mae':float(abs(e).mean()),'trajectory_rmse':float(np.sqrt((e**2).mean())),
                 'trajectory_direction_accuracy':float(((p>0)==(y>0)).mean()),
                 'signal_mae':m['point']['mae'],'signal_rmse':m['point']['rmse'],
                 'direction_accuracy':m['direction']['accuracy'],'balanced_accuracy':m['direction']['balanced_accuracy'],
                 'ic':m['cross_section']['ic']['mean'],'rankic':m['cross_section']['rankic']['mean'],
                 'net_return':float(np.prod(1+dd.top_net)-1),'sharpe':m['portfolio']['sharpe'],
                 'max_drawdown':m['portfolio']['max_drawdown'],'turnover':m['portfolio']['turnover'],
                 'q5_q1':m['portfolio']['q5_q1'],'mean_top_excess':m['portfolio']['top_excess']})
        for row in result['metrics']['trajectory_by_horizon']:
            horizons.append({'architecture':arch,'seed':seed,'phase':phase,**row})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'comparison_metrics.csv',index=False)
    numeric=summary.select_dtypes('number').columns.difference(['seed'])
    summary.groupby(['architecture','phase','signal'])[numeric].agg(['mean','std']).to_csv(out/'seed_summary.csv')
    pd.DataFrame(horizons).to_csv(out/'metrics_by_horizon.csv',index=False)
    paired=[]
    # Seeds share dates; average paired differences over seeds before date-block bootstrap.
    rng=np.random.default_rng(20261104)
    for phase in ('static','online'):
        for mode in ('path','endpoint'):
            for left,right in [('temporal_attention','adapter'),('temporal_attention','temporal_mean')]:
                aa=[daily[left,s,phase,mode] for s in args.seeds];bb=[daily[right,s,phase,mode] for s in args.seeds]
                n=len(aa[0])
                for block in (10,20):
                    starts=rng.integers(0,n-block+1,size=(3000,math.ceil(n/block)))
                    draws=(starts[:,:,None]+np.arange(block)).reshape(3000,-1)[:,:n]
                    for metric in ('mae','mse','direction','ic','rankic','top_net','turnover'):
                        diff=np.mean([a[metric].to_numpy()-b[metric].to_numpy() for a,b in zip(aa,bb)],axis=0)
                        boot=diff[draws].mean(1)
                        paired.append({'phase':phase,'signal':mode,'comparison':left+'_minus_'+right,'metric':metric,'block':block,
                                       'delta':float(diff.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975))})
    pd.DataFrame(paired).to_csv(out/'paired_date_bootstrap.csv',index=False)
    return summary


def main(args):
    started=time.perf_counter();deadline=started+args.budget_seconds
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    meta_path=ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv'
    meta=pd.read_csv(meta_path,dtype={'stock':str})
    rows,cohort=select_panel(meta,args.stocks,args.train_dates,args.val_dates,args.test_dates)
    rows.to_csv(out/'selected_samples.csv',index=False)
    config={**vars(args),'output_dir':str(out),'cohort':cohort,'architecture':ARCHITECTURES,'attention_width':32,'heads':2,
            'selection':'validation MSE only; include original step0','split_rows':rows.groupby('split').size().to_dict(),
            'portfolio':'subset Top6/Drop1, minhold5, one-way15bps; not full CSI300 Top30',
            'retrospective':True,'protocol':'historical offline refinement then separate static and complete-feedback online streams'}
    write_json(out/'run_config.json',config)
    source={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/kronos_continual/attention_adapter.py',ROOT/'scripts/experiments/run_online_module_pilot.py',ROOT/'evaluate/metrics.py']}
    write_json(out/'source_manifest.json',source)
    h,cache_info=build_sequences(rows,out,args,deadline)
    checkpoint=ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt'
    state=torch.load(checkpoint,weights_only=True);models={(a,s):load_model(a,state,s) for s in args.seeds for a in ARCHITECTURES}
    labels=read_npy_rows(ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2/y.npy',rows.row_id.to_numpy()).astype(np.float32)
    test_ix=np.flatnonzero(rows.split.eq('test'))
    initial=predict(models['adapter',args.seeds[0]],'adapter',h,test_ix)
    for (a,s),m in models.items():np.testing.assert_array_equal(predict(m,a,h,test_ix),initial)
    np.save(out/'initial_test_predictions.npy',initial)
    best,epochs=train_offline(models,rows,h,labels,out,args,deadline)
    frames={}
    for (a,s),m in models.items():
        pred=predict(m,a,h,test_ix)
        frames[a,s,'static']=prediction_frame(rows,test_ix,pred,labels,a,s,'static')
    frames.update(online_predictions(models,rows,h,labels,out,args))
    summary=summarize(frames,out,args)
    write_json(out/'status.json',{'status':'complete','seconds':time.perf_counter()-started,'epochs':epochs,
                                'rows':len(rows),'test_rows':len(test_ix),'test_dates':rows[rows.split.eq('test')].as_of_date.nunique(),
                                'checkpoint_sha256':sha256(checkpoint),'meta_sha256':sha256(meta_path)})
    print(summary.to_string(index=False),flush=True);print('COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stocks',type=int,default=60);p.add_argument('--train-dates',type=int,default=80)
    p.add_argument('--val-dates',type=int,default=40);p.add_argument('--test-dates',type=int,default=80)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seeds',type=int,nargs='+',default=[100,200,300]);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-seconds',type=int,default=3600)
    args=p.parse_args()
    try:main(args)
    except Exception as exc:
        if args.output_dir.exists():write_json(args.output_dir/'FAILED.json',{'error':repr(exc)})
        raise
