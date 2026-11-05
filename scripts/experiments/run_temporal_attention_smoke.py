"""Real-data CPU pilot: 16 held-out queries, separate historical support/validation."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.runtime/torchinductor_cache'))
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/hf'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.kronos_continual.attention_adapter import TemporalAttentionModel
from scripts.data.build_continual_csi300_hidden_cache import load_frames, normalise_ohlcv, stamps
from scripts.experiments.run_delayed_feedback_pilot import Model, sha256


def select_rows(meta):
    # Fixed dates/stocks, selected solely by metadata, before reading targets.
    chosen = {}
    common = set.intersection(*[set(meta.loc[meta.split.eq(s), 'stock']) for s in ('train','val','test')])
    stocks = sorted(common)[:4]
    for split, n_dates in [('train',8),('val',4),('test',4)]:
        part = meta[meta.split.eq(split) & meta.stock.isin(stocks)]
        dates = sorted(d for d,g in part.groupby('as_of_date') if len(g)==4)
        # Use the last training dates; spread validation and test dates across their periods.
        selected = dates[-n_dates:] if split == 'train' else [dates[i] for i in np.linspace(0,len(dates)-1,n_dates,dtype=int)]
        chosen[split] = part[part.as_of_date.isin(selected)].sort_values(['as_of_date','stock']).copy()
    assert chosen['train'].label_end_date.max() < chosen['val'].as_of_date.min()
    assert chosen['val'].label_end_date.max() < chosen['test'].as_of_date.min()
    return chosen


def read_npy_rows(path, ids):
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f))
        assert not fortran
        offset = f.tell()
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        values = []
        for i in ids:
            f.seek(offset + int(i) * stride)
            values.append(np.frombuffer(f.read(stride), dtype=dtype).copy().reshape(shape[1:]))
    return np.stack(values)


def metrics(p,y):
    e=p-y
    return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),
            'per_horizon_mae':np.abs(e).mean(0).tolist(),'per_horizon_rmse':np.sqrt(np.square(e).mean(0)).tolist()}


def main(out):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.perf_counter(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    cache=ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2'
    meta=pd.read_csv(cache/'meta.csv',dtype={'stock':str}); chosen=select_rows(meta)
    # Only 64 real windows are encoded: 32 support, 16 validation, 16 held-out query.
    rows=pd.concat([chosen[s] for s in ('train','val','test')],ignore_index=True)
    rows.to_csv(out/'selected_samples.csv',index=False)
    windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str).drop_duplicates('as_of_date').set_index('as_of_date')
    frames=load_frames(ROOT/'data/csi300_daily/csi300_daily')
    sys.path.insert(0,str(ROOT/'third_party/Kronos_official_src/Kronos-master'))
    from model import Kronos,KronosTokenizer
    hub=ROOT/'.runtime/hf/hub'
    model_dir=next((hub/'models--NeoQuasar--Kronos-small/snapshots').iterdir())
    tok_dir=next((hub/'models--NeoQuasar--Kronos-Tokenizer-base/snapshots').iterdir())
    backbone=Kronos.from_pretrained(str(model_dir)).eval(); tok=KronosTokenizer.from_pretrained(str(tok_dir)).eval()
    for m in (backbone,tok): m.requires_grad_(False)
    frozen_before={n:sha256(d/'model.safetensors') for n,d in [('backbone',model_dir),('tokenizer',tok_dir)]}
    xs=[]; ts=[]
    for row in rows.itertuples():
        w=windows.loc[row.as_of_date]
        dates=pd.to_datetime(w.input_dates.split('|'))
        assert len(dates)==90 and str(dates[-1].date())==row.as_of_date
        xs.append(normalise_ohlcv(frames[row.stock].loc[dates])); ts.append(stamps(dates))
    extract_start=time.perf_counter(); seq=[]
    with torch.no_grad():
        for i in range(0,len(rows),16):
            token=tok.encode(torch.tensor(np.stack(xs[i:i+16])),half=True)
            _,h=backbone.decode_s1(token[0],token[1],torch.tensor(np.stack(ts[i:i+16])))
            seq.append(h.float().cpu())
    h=torch.cat(seq); extract_seconds=time.perf_counter()-extract_start
    assert h.shape==(64,90,512) and torch.isfinite(h).all()
    checkpoint=ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt'
    base=Model().eval(); base.load_state_dict(torch.load(checkpoint,weights_only=True))
    # Preserve float32 after measuring fp16's initial prediction error.
    with torch.no_grad():
        fp16_delta=float((base(h[:,-1])-base(h.half().float()[:,-1])).abs().max())
    historical={}
    for split in ('train','test'):
        old_path=ROOT/f'experiments/csi300_aligned/reference/hidden_cache/hidden_{split}.npy'
        split_meta=meta[meta.split.eq(split)].reset_index(drop=True)
        index={(r.stock,r.as_of_date):i for i,r in enumerate(split_meta.itertuples())}
        pos=np.flatnonzero(rows.split.eq(split))
        with open(old_path, 'rb') as f:
            version=np.lib.format.read_magic(f)
            shape,_,dtype=np.lib.format.read_array_header_1_0(f)
            expected_bytes=f.tell()+int(np.prod(shape))*dtype.itemsize
        if old_path.stat().st_size != expected_bytes:
            historical[split]={'status':'incomplete_legacy_cache','expected_bytes':expected_bytes,'actual_bytes':old_path.stat().st_size}
            continue
        expected=read_npy_rows(old_path,[index[(rows.iloc[i].stock,rows.iloc[i].as_of_date)] for i in pos])
        historical[split]={'max_abs_hidden_error':float(np.max(np.abs(h[pos,-1].numpy()-expected)))}
    for split in ('train','val','test'):
        mask=rows.split.eq(split).to_numpy(); np.save(out/f'hidden_seq_{split}.npy',h[mask].numpy())
    y_all=None
    y=torch.tensor(read_npy_rows(cache/'y.npy', rows.row_id.to_numpy()),dtype=torch.float32)
    ids={s:np.flatnonzero(rows.split.eq(s)) for s in ('train','val','test')}
    results=[]; histories=[]; predictions=[]; diagnostics=[]
    for seed in (100,200,300):
        for arch in ('adapter','temporal_mean','temporal_attention'):
            torch.manual_seed(seed)
            if arch=='adapter':
                model=copy.deepcopy(base)
            else:
                model=TemporalAttentionModel(mode=arch).eval()
                model.adapter.load_state_dict(base.adapter.state_dict()); model.head.load_state_dict(base.head.state_dict())
            def forward(ix):
                x=h[ix]
                if arch=='adapter': return model(x[:,-1])
                return model(x)
            with torch.no_grad():
                initial_error=float((forward(ids['test'])-base(h[ids['test'],-1])).abs().max())
            assert initial_error==0
            groups=[{'params':model.head.parameters(),'lr':1e-4},{'params':model.adapter.parameters(),'lr':1e-5}]
            if arch!='adapter': groups.append({'params':[p for p in model.pool.parameters() if p.requires_grad],'lr':1e-5})
            opt=torch.optim.AdamW(groups,weight_decay=0)
            with torch.no_grad(): best=float((forward(ids['val'])-y[ids['val']]).square().mean())
            best_step=0; best_state=copy.deepcopy(model.state_dict()); train_start=time.perf_counter(); gradient_checks={}
            for step in range(1,101):
                opt.zero_grad(); loss=torch.nn.functional.smooth_l1_loss(forward(ids['train']),y[ids['train']]); loss.backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(),name
                        if step==2: gradient_checks[name]=float(p.grad.norm())
                nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
                if step in (1,10,25,50,100):
                    with torch.no_grad(): val=float((forward(ids['val'])-y[ids['val']]).square().mean())
                    histories.append({'seed':seed,'architecture':arch,'step':step,'train_loss':float(loss.detach()),'validation_mse':val})
                    if val<best: best=val; best_step=step; best_state=copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            with torch.no_grad(): pred=forward(ids['test']).numpy()
            name=f'{arch}_seed{seed}'; torch.save(model.state_dict(),out/f'{name}.pt')
            restored=copy.deepcopy(model); restored.load_state_dict(torch.load(out/f'{name}.pt',weights_only=True))
            with torch.no_grad():
                restored_pred=restored(h[ids['test'],-1]) if arch=='adapter' else restored(h[ids['test']])
                torch.testing.assert_close(restored_pred,torch.from_numpy(pred),rtol=0,atol=0)
            results.append({'seed':seed,'architecture':arch,'selected_step':best_step,'validation_mse':best,'initial_max_error':initial_error,'train_seconds':time.perf_counter()-train_start,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),**metrics(pred,y[ids['test']].numpy())})
            for j,i in enumerate(ids['test']):
                predictions.append({'seed':seed,'architecture':arch,'row_id':int(rows.iloc[i].row_id),'as_of_date':rows.iloc[i].as_of_date,'stock':rows.iloc[i].stock,**{f'pred_return_{k+1}':float(pred[j,k]) for k in range(10)},**{f'actual_return_{k+1}':float(y[i,k]) for k in range(10)}})
            if arch=='temporal_attention':
                with torch.no_grad(): _,weights=model.pool(h[ids['test']],return_weights=True)
                diagnostics.append({'seed':seed,'mean_attention_weights':weights.mean(0).tolist(),'gradient_norms_step2':gradient_checks})
            print(json.dumps({k:v for k,v in results[-1].items() if not isinstance(v,list)}),flush=True)
    assert all(p.grad is None and not p.requires_grad for m in (backbone,tok) for p in m.parameters())
    report={'status':'complete','protocol':'offline pilot; 32 train / 16 validation / 16 held-out queries; no online adaptation','seeds':[100,200,300],'selection':'lowest validation MSE, including step0; test never used for selection','feature_extraction_seconds':extract_seconds,'total_seconds':time.perf_counter()-start,'sequence_shape':list(h.shape),'dtype':'float32','fp16_initial_prediction_max_error':fp16_delta,'historical_cache_alignment':historical,'checkpoint_sha256':sha256(checkpoint),'weight_sha256':frozen_before,'frozen_backbone_no_grad':True,'results':results,'limitation':'Only 4 test dates x 4 fixed stocks. Exploratory; no stable advantage, full-226-day or portfolio conclusion is justified.'}
    pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False); pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    (out/'attention_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); main(a.output_dir)
