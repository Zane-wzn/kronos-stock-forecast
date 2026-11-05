"""CPU smoke for Adapter -> native coarse/fine tokens -> official K-line decoder.

This is an implementation/connectivity check, not a performance claim.
"""
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
os.environ.setdefault('HF_HOME', str(ROOT/'.runtime/hf'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT/'.runtime/torchinductor_cache'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.kronos_continual.token_adapter import TokenPredictionAdapter, shifted_future_loss, normalize_history_future
from scripts.experiments.run_temporal_attention_smoke import select_rows
from scripts.experiments.run_delayed_feedback_pilot import sha256
from scripts.experiments.run_temporal_attention_pilot import write_json, PCOLS, ACOLS
from scripts.data.build_continual_csi300_hidden_cache import load_frames, stamps


def frozen_hash(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def token_metrics(model, tokens, time_features, ids):
    # Fixed coarse sampling seed per validation call: paired, reproducible CE.
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(20261105)
        total=[]
        for start in range(0,len(ids),8):
            ix=ids[start:start+8];s1,s2=tokens[0][ix],tokens[1][ix]
            l1,l2=model(s1[:,:-1],s2[:,:-1],time_features[ix,:-1],fine_start=89)
            _,c1,c2=shifted_future_loss(l1,l2,s1,s2,90)
            total.append([len(ix),float(c1),float(c2),float((l1[:,89:].argmax(-1)==s1[:,90:]).float().mean()),float((l2[:,89:].argmax(-1)==s2[:,90:]).float().mean())])
    a=np.array(total);w=a[:,0]/a[:,0].sum();v=(a[:,1:]*w[:,None]).sum(0)
    return {'ce':float((v[0]+v[1])/2),'coarse_ce':float(v[0]),'fine_ce_sampled_coarse':float(v[1]),'coarse_accuracy':float(v[2]),'fine_accuracy_sampled_coarse':float(v[3])}


def main(args):
    started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(100)
    out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    sys.path.insert(0,str(ROOT/'third_party/Kronos_official_src/Kronos-master'))
    from model import Kronos,KronosTokenizer,KronosPredictor
    hub=ROOT/'.runtime/hf/hub'
    model_dir=sorted((hub/'models--NeoQuasar--Kronos-small/snapshots').iterdir())[0]
    tok_dir=sorted((hub/'models--NeoQuasar--Kronos-Tokenizer-base/snapshots').iterdir())[0]
    base=Kronos.from_pretrained(str(model_dir)).eval().requires_grad_(False)
    tokenizer=KronosTokenizer.from_pretrained(str(tok_dir)).eval().requires_grad_(False)
    hashes={'backbone':frozen_hash(base),'tokenizer':frozen_hash(tokenizer)}
    meta=pd.read_csv(ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv',dtype={'stock':str})
    chosen=select_rows(meta);rows=pd.concat([chosen[s] for s in ('train','val','test')],ignore_index=True)
    rows.to_csv(out/'selected_samples.csv',index=False)
    windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str).drop_duplicates('as_of_date').set_index('as_of_date')
    frames=load_frames(ROOT/'data/csi300_daily/csi300_daily')
    xs=[];ts=[];history=[];future=[];input_dates=[];target_dates=[]
    fields=['open','high','low','close','volume','amount']
    for row in rows.itertuples():
        w=windows.loc[row.as_of_date];past_dates=pd.to_datetime(w.input_dates.split('|'));next_dates=pd.to_datetime(w.label_dates.split('|'))
        past=frames[row.stock].loc[past_dates,fields].copy();fut=frames[row.stock].loc[next_dates,fields].copy()
        assert len(past)==90 and len(fut)==10 and str(past_dates[-1].date())==row.as_of_date and next_dates[0]>past_dates[-1]
        norm,_,_=normalize_history_future(past.to_numpy(),fut.to_numpy())
        xs.append(norm);ts.append(stamps(past_dates.append(next_dates)));history.append(past);future.append(fut)
        input_dates.append(pd.Series(past_dates));target_dates.append(pd.Series(next_dates))
    x=torch.from_numpy(np.stack(xs));time_features=torch.from_numpy(np.stack(ts))
    with torch.no_grad():
        tokens=tokenizer.encode(x,half=True)
        # Token labels must not change historical context or target earlier tokens.
        changed=x.clone();changed[:,95:]=-changed[:,95:]
        altered=tokenizer.encode(changed,half=True)
        assert all(torch.equal(a[:,:95],b[:,:95]) for a,b in zip(tokens,altered))
    torch.save({'coarse':tokens[0],'fine':tokens[1],'timestamps':time_features},out/'tokens.pt')
    ids={s:np.flatnonzero(rows.split.eq(s)) for s in ('train','val','test')}
    config={'history_length':90,'future_length':10,'init_modes':['identity','return_checkpoint'],
            'trainable':'Adapter only','trainable_parameters':67136,'loss':'mean CE(coarse, fine conditioned on sampled predicted coarse); future positions only',
            'coarse_teacher_forcing':False,'temporal_teacher_forcing':'historical prefixes including earlier future target tokens during training; free-running rollout at evaluation',
            'fine_dependency':'native decode_s2 on each prefix; identical single-query RoPE semantics','sampling':{'T':.6,'top_p':.9,'top_k':0,'sample_count':args.sample_count},
            'steps':args.steps,'learning_rate':1e-5,'batch_size':8,'train_rows':len(ids['train']),'validation_rows':len(ids['val']),'test_rows':len(ids['test']),
            'checkpoint_selection':'fixed validation sampled-coarse CE; include step0; evaluate every5steps','smoke_only':True,
            'model_snapshot':model_dir.name,'tokenizer_snapshot':tok_dir.name}
    write_json(out/'run_config.json',config)
    write_json(out/'source_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/kronos_continual/token_adapter.py',ROOT/'third_party/Kronos_official_src/Kronos-master/model/kronos.py',ROOT/'third_party/Kronos_official_src/Kronos-master/model/module.py']})
    checkpoint=ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt'
    learned={};logs=[];results=[];gradient_checks={}
    for mode in ('identity','return_checkpoint'):
        torch.manual_seed(100);model=TokenPredictionAdapter(copy.deepcopy(base)).eval()
        if mode=='return_checkpoint':model.load_return_adapter(checkpoint)
        v=token_metrics(model,tokens,time_features,ids['val']);initial_ce=v['ce'];best=v['ce'];best_step=0;best_state=copy.deepcopy(model.adapter.state_dict())
        results.append({'mode':mode,'stage':'initial_validation',**v})
        opt=torch.optim.AdamW(model.adapter.parameters(),lr=1e-5,weight_decay=0)
        rng=np.random.default_rng(100)
        for step in range(1,args.steps+1):
            ix=rng.choice(ids['train'],8,replace=False);s1,s2=tokens[0][ix],tokens[1][ix]
            opt.zero_grad(set_to_none=True)
            torch.manual_seed(100+step)
            l1,l2=model(s1[:,:-1],s2[:,:-1],time_features[ix,:-1],fine_start=89)
            loss,c1,c2=shifted_future_loss(l1,l2,s1,s2,90);loss.backward()
            assert all(p.grad is None for p in model.backbone.parameters())
            if step==2:gradient_checks[mode]={n:float(p.grad.norm()) for n,p in model.adapter.named_parameters()}
            nn.utils.clip_grad_norm_(model.adapter.parameters(),1);opt.step()
            if step%5==0 or step==args.steps:
                v=token_metrics(model,tokens,time_features,ids['val'])
                if v['ce']<best:best=v['ce'];best_step=step;best_state=copy.deepcopy(model.adapter.state_dict())
                row={'mode':mode,'step':step,'train_ce':float(loss.detach()),'coarse_ce':float(c1.detach()),'fine_ce':float(c2.detach()),**{'val_'+k:z for k,z in v.items()}}
                logs.append(row);print(json.dumps(row),flush=True)
        model.adapter.load_state_dict(best_state)
        path=out/f'checkpoints/{mode}.pt';torch.save({'adapter':best_state,'selected_step':best_step,'model_snapshot':model_dir.name,'tokenizer_snapshot':tok_dir.name,'initialization':mode},path)
        model.adapter.load_state_dict(torch.load(path,weights_only=True)['adapter'])
        learned[mode]=model
        results.append({'mode':mode,'stage':'selected_test_token', 'selected_step':best_step,'initial_val_ce':initial_ce,'selected_val_ce':best,**token_metrics(model,tokens,time_features,ids['test'])})
    # Initial identity must reproduce the full native generation for matched RNG.
    identity=TokenPredictionAdapter(copy.deepcopy(base)).eval()
    raw_predictor=KronosPredictor(base,tokenizer,device='cpu')
    identity_predictor=KronosPredictor(identity,tokenizer,device='cpu')
    ix=ids['test'][:1]
    kwargs={'pred_len':10,'T':.6,'top_k':0,'top_p':.9,'sample_count':args.sample_count,'verbose':False}
    torch.manual_seed(123)
    a=raw_predictor.predict_batch([history[i] for i in ix],[input_dates[i] for i in ix],[target_dates[i] for i in ix],**kwargs)
    torch.manual_seed(123)
    b=identity_predictor.predict_batch([history[i] for i in ix],[input_dates[i] for i in ix],[target_dates[i] for i in ix],**kwargs)
    identity_error=float(abs(a[0].to_numpy()-b[0].to_numpy()).max());assert identity_error==0
    predictions=[];point_metrics=[]
    models={'official_zero_shot':base,**learned}
    for name,model in models.items():
        predictor=KronosPredictor(model,tokenizer,device='cpu')
        for i in ids['test']:
            # Same per-stock/date seed and sampling parameters across models.
            seed=int.from_bytes(hashlib.sha256((str(rows.iloc[i].stock)+'|'+str(rows.iloc[i].as_of_date)).encode()).digest()[:4],'little')
            torch.manual_seed(seed)
            pp=predictor.predict_batch([history[i]],[input_dates[i]],[target_dates[i]],**kwargs)[0]
            close_t=float(history[i]['close'].iloc[-1]);pc=pp['close'].to_numpy();ac=future[i]['close'].to_numpy()
            if not np.isfinite(pc).all():raise FloatingPointError('nonfinite generated price')
            rec={'method':name,'as_of_date':rows.iloc[i].as_of_date,'stock':rows.iloc[i].stock,'current_close':close_t,
                 **{f'pred_close_{k+1}':float(pc[k]) for k in range(10)},**{f'pred_return_{k+1}':float(pc[k]/close_t-1) for k in range(10)},
                 **{f'actual_return_{k+1}':float(ac[k]/close_t-1) for k in range(10)}}
            predictions.append(rec)
        f=pd.DataFrame([r for r in predictions if r['method']==name]);e=f[PCOLS].to_numpy()-f[ACOLS].to_numpy()
        point_metrics.append({'method':name,'trajectory_mae':float(abs(e).mean()),'trajectory_rmse':float(np.sqrt((e**2).mean())),
                              'nonpositive_predicted_close_count':int((f[[f'pred_close_{k}' for k in range(1,11)]].to_numpy()<=0).sum())})
    assert hashes=={'backbone':frozen_hash(base),'tokenizer':frozen_hash(tokenizer)}
    assert all(frozen_hash(m.backbone)==hashes['backbone'] for m in learned.values())
    pd.DataFrame(logs).to_csv(out/'training_history.csv',index=False);pd.DataFrame(results).to_csv(out/'token_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False);pd.DataFrame(point_metrics).to_csv(out/'generation_metrics.csv',index=False)
    write_json(out/'audit.json',{'frozen_hashes':hashes,'unchanged':True,'identity_official_rollout_max_error':identity_error,'tokenizer_causal_perturbation':True,'adapter_gradient_norms_step2':gradient_checks,'input_normalization':'history90 only; no future statistics','checkpoint_sha256':sha256(checkpoint)})
    write_json(out/'status.json',{'status':'complete','seconds':time.perf_counter()-started,'smoke_only':True,'token_metrics':results,'generation_metrics':point_metrics,
                               'limitation':'16 test paths / 4 dates / one seed; connectivity and numerical checks, no reliable effectiveness conclusion'})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--sample-count',type=int,default=2);args=p.parse_args();main(args)
