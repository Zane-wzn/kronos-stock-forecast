"""CPU four-arm LoRA x objective pilot, with fixed validation and two test blocks."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT/'.runtime/hf'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.runtime/torchinductor_cache'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.experiments.run_temporal_attention_pilot import select_panel,write_json,PCOLS,ACOLS,KEYS,prediction_frame
from scripts.experiments.run_temporal_attention_smoke import read_npy_rows
from scripts.experiments.run_delayed_feedback_pilot import Model,sha256
from scripts.data.build_continual_csi300_hidden_cache import normalise_ohlcv,stamps
from src.kronos_continual.lora_suffix import KronosSuffix,objective
from evaluate.metrics import evaluate_frame,EvaluationConfig

ARMS={'A0':(False,0.),'A1':(True,0.),'B0':(False,.1),'B1':(True,.1)}


def state_hash(model, frozen_only=False):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not frozen_only or not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(out,args):
    meta_path=ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv'
    meta=pd.read_csv(meta_path,dtype={'stock':str})
    rows,stocks=select_panel(meta,60,80,40,160)
    rows.to_csv(out/'selected_samples.csv',index=False)
    ys=read_npy_rows(ROOT/'experiments/csi300_aligned/doubleadapt_cache_v2/y.npy',rows.row_id).astype('float32')
    np.save(out/'labels.npy',ys)
    hub=ROOT/'.runtime/hf/hub'
    model_dir=sorted((hub/'models--NeoQuasar--Kronos-small/snapshots').iterdir())[0]
    tok_dir=sorted((hub/'models--NeoQuasar--Kronos-Tokenizer-base/snapshots').iterdir())[0]
    sys.path.insert(0,str(ROOT/'third_party/Kronos_official_src/Kronos-master'))
    from model import Kronos,KronosTokenizer
    backbone=Kronos.from_pretrained(str(model_dir)).eval().requires_grad_(False)
    tok=KronosTokenizer.from_pretrained(str(tok_dir)).eval().requires_grad_(False)
    before=(state_hash(backbone),state_hash(tok))
    windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str,usecols=['as_of_date','input_dates']).drop_duplicates('as_of_date').set_index('as_of_date')
    frames={};raw_hash={}
    for stock in stocks:
        path=ROOT/'data/csi300_daily/csi300_daily'/(stock.replace('.','_')+'.csv')
        f=pd.read_csv(path);f['date']=pd.to_datetime(f.date)
        if f.date.duplicated().any():raise AssertionError('duplicate dates')
        f['amount']=f[['open','high','low','close']].mean(axis=1)*f.volume
        frames[stock]=f.set_index('date');raw_hash[stock]=sha256(path)
    (out/'cache').mkdir()
    prefix=np.lib.format.open_memmap(out/'cache/prefix.npy',mode='w+',dtype='float32',shape=(len(rows),90,512))
    hidden=np.lib.format.open_memmap(out/'cache/hidden.npy',mode='w+',dtype='float32',shape=(len(rows),512))
    start=time.perf_counter();done=0;reference_max=0.
    # Original final cache from previous pilot is only used for SAME-key identity check.
    old_root=ROOT/'experiments/csi300_aligned/models/temporal_attention_pilot_60x80_20261104'
    old_meta=pd.read_csv(old_root/'selected_samples.csv',dtype={'stock':str})
    old_lookup={(r.as_of_date,r.stock):i for i,r in enumerate(old_meta.itertuples())}
    old_hidden=np.load(old_root/'cache/hidden_seq.npy',mmap_mode='r')
    with torch.no_grad():
        for (split,date),g in rows.groupby(['split','as_of_date'],sort=False):
            dates=pd.to_datetime(windows.loc[date,'input_dates'].split('|'))
            assert len(dates)==90 and str(dates[-1].date())==date
            for begin in range(0,len(g),16):
                part=g.iloc[begin:begin+16];x=np.stack([normalise_ohlcv(frames[s].loc[dates]) for s in part.stock]);ts=np.repeat(stamps(dates)[None],len(part),axis=0)
                tokens=tok.encode(torch.from_numpy(x),half=True)
                z=backbone.token_drop(backbone.embedding([tokens[0],tokens[1]])+backbone.time_emb(torch.from_numpy(ts)))
                for layer in backbone.transformer[:6]:z=layer(z)
                prefix[part.index]=z.numpy()
                for layer in backbone.transformer[6:]:z=layer(z)
                last=backbone.norm(z)[:,-1];hidden[part.index]=last.numpy()
                for j,r in enumerate(part.itertuples()):
                    key=(r.as_of_date,r.stock)
                    if key in old_lookup:reference_max=max(reference_max,float(np.max(abs(last[j].numpy()-old_hidden[old_lookup[key],-1]))))
            done+=len(g)
            if done%1000<len(g) or done==len(rows):
                prefix.flush();hidden.flush();rec={'stage':'prefix_cache','rows':done,'total':len(rows),'seconds':time.perf_counter()-start}
                write_json(out/'progress.json',rec);print(json.dumps(rec),flush=True)
    assert before==(state_hash(backbone),state_hash(tok))
    info={'shape':list(prefix.shape),'dtype':'float32','seconds':time.perf_counter()-start,'model_snapshot':model_dir.name,'tokenizer_snapshot':tok_dir.name,'backbone_tensor_sha256':before[0],'tokenizer_tensor_sha256':before[1],'prefix_sha256':sha256(out/'cache/prefix.npy'),'hidden_sha256':sha256(out/'cache/hidden.npy'),'meta_sha256':sha256(meta_path),'raw_sha256':raw_hash,'previous_same_key_last_hidden_max_error':reference_max}
    write_json(out/'cache_manifest.json',info)
    suffix=(copy.deepcopy(list(backbone.transformer[6:])),copy.deepcopy(backbone.norm))
    del backbone,tok,frames,old_hidden
    return rows,ys,prefix,hidden,suffix


def predict(model,lora,prefix,hidden,ids):
    chunks=[]
    with torch.no_grad():
        for i in range(0,len(ids),32):
            ix=ids[i:i+32]
            if lora:p=model(torch.from_numpy(np.array(prefix[ix])))
            else:p=model(torch.from_numpy(np.array(hidden[ix])))
            chunks.append(p.numpy())
    return np.concatenate(chunks)


def validation(p,y,dates):
    vals=[]
    for d in sorted(set(dates)):
        mask=dates==d;ps=pd.Series(p[mask].mean(1)).rank().to_numpy();ys=pd.Series(y[mask].mean(1)).rank().to_numpy()
        if np.std(ps)>0 and np.std(ys)>0:vals.append(float(np.corrcoef(ps,ys)[0,1]))
    return {'mae':float(abs(p-y).mean()),'rmse':float(np.sqrt(((p-y)**2).mean())),'mse':float(((p-y)**2).mean()),'rankic':float(np.mean(vals))}


def run(args):
    started=time.perf_counter();out=args.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    config={**vars(args),'output_dir':str(out),'arms':ARMS,'lora':{'layers':[6,7],'targets':['q_proj','v_proj'],'rank':4,'alpha':8},'seeds':[100,200,300],'batch':'one origin date, max32 stocks','selection':'validation Path RankIC subject to <=1% relative MAE/RMSE deterioration vs original checkpoint; fallback epoch0','online':False,'test_blocks':'first80 and second80 observed test dates, fixed before label access','retrospective':True}
    write_json(out/'run_config.json',config)
    rows,y,prefix,hidden,suffix=prepare(out,args)
    state=torch.load(ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt',weights_only=True)
    tr=np.flatnonzero(rows.split.eq('train'));va=np.flatnonzero(rows.split.eq('val'));te=np.flatnonzero(rows.split.eq('test'))
    scales=torch.tensor(np.maximum(y[tr].std(0),1e-4));score_scale=float(max(y[tr].mean(1).std(),1e-4))
    write_json(out/'loss_scales.json',{'horizon':scales.tolist(),'score':score_scale,'fit':'train only'})
    base=Model().eval();base.load_state_dict(state)
    baseline=validation(predict(base,False,prefix,hidden,va),y[va],rows.iloc[va].as_of_date.to_numpy())
    write_json(out/'validation_reference.json',baseline)
    histories=[];selections=[];frozen_checks=[];gradient_checks=[];evaluations=[]
    train_groups=[g.index.to_numpy() for _,g in rows[rows.split.eq('train')].groupby('as_of_date')]
    # Work is paired by equal number of optimizer steps, not equal wall time.
    for seed in (100,200,300):
        for arm,(lora,weight) in ARMS.items():
            torch.manual_seed(seed);head=Model().eval();head.load_state_dict(state)
            if lora:model=KronosSuffix(copy.deepcopy(suffix[0]),copy.deepcopy(suffix[1]),head,True).eval()
            else:model=head
            frozen_hash=state_hash(model,True)
            groups=[{'params':model.head.parameters(),'lr':1e-4},{'params':model.adapter.parameters(),'lr':1e-5}]
            if lora:groups.append({'params':[p for n,p in model.named_parameters() if n.endswith('.a') or n.endswith('.b')],'lr':1e-5})
            opt=torch.optim.AdamW(groups,weight_decay=0)
            first=va[:16];np.testing.assert_allclose(predict(model,lora,prefix,hidden,first),predict(base,False,prefix,hidden,first),atol=1e-6,rtol=0)
            best_rank=baseline['rankic'];best_epoch=0;best=copy.deepcopy(model.state_dict());arm_start=time.perf_counter();step=0
            for epoch in range(1,args.epochs+1):
                losses=[];paths=[];ranks=[]
                order=np.random.default_rng(seed+epoch).permutation(len(train_groups))
                for group_id in order:
                    ix=train_groups[group_id]
                    if len(ix)>32:ix=np.random.default_rng(seed+epoch*1000+int(group_id)).choice(ix,32,replace=False)
                    opt.zero_grad(set_to_none=True)
                    x=torch.from_numpy(np.array(prefix[ix] if lora else hidden[ix]));target=torch.from_numpy(y[ix])
                    pred=model(x);loss,pl,rl=objective(pred,target,scales,score_scale,weight);loss.backward()
                    for n,p in model.named_parameters():
                        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):raise AssertionError(n)
                    if lora and step==1:gradient_checks.append({'arm':arm,'seed':seed,'grads':{n:float(p.grad.norm()) for n,p in model.named_parameters() if n.endswith('.a') or n.endswith('.b')}})
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step();step+=1
                    losses.append(float(loss.detach()));paths.append(float(pl));ranks.append(float(rl))
                v=validation(predict(model,lora,prefix,hidden,va),y[va],rows.iloc[va].as_of_date.to_numpy())
                eligible=v['mae']<=baseline['mae']*1.01 and v['rmse']<=baseline['rmse']*1.01
                if eligible and v['rankic']>best_rank:best_rank=v['rankic'];best_epoch=epoch;best=copy.deepcopy(model.state_dict())
                rec={'arm':arm,'seed':seed,'epoch':epoch,'steps':step,'loss':float(np.mean(losses)),'path_loss':float(np.mean(paths)),'rank_loss':float(np.mean(ranks)),'eligible':eligible,**{'val_'+k:v for k,v in v.items()},'seconds':time.perf_counter()-arm_start}
                histories.append(rec);pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False);print(json.dumps(rec),flush=True)
            assert frozen_hash==state_hash(model,True)
            frozen_checks.append({'arm':arm,'seed':seed,'frozen_hash':frozen_hash,'unchanged':True})
            model.load_state_dict(best);ckpt=out/f'checkpoints/{arm}_{seed}.pt';torch.save(model.state_dict(),ckpt)
            reload_state=torch.load(ckpt,weights_only=True);model.load_state_dict(reload_state)
            selections.append({'arm':arm,'seed':seed,'epoch':best_epoch,'steps_if_selected':best_epoch*len(train_groups),'validation_rankic':best_rank,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':time.perf_counter()-arm_start})
            # Test only after selection, for two fixed blocks; never select using test metrics.
            pred=predict(model,lora,prefix,hidden,te)
            frame=rows.iloc[te][KEYS+['row_id']].copy();frame[PCOLS]=pred;frame[ACOLS]=y[te];frame['pred_signal']=pred.mean(1);frame['actual_signal']=y[te].mean(1);frame['pred_endpoint_return']=pred[:,-1];frame['actual_endpoint_return']=y[te,-1];frame['method']=arm;frame['seed']=seed
            td=sorted(frame.as_of_date.unique())
            for block,dates in [('block1',td[:80]),('block2',td[80:])]:
                f=frame[frame.as_of_date.isin(dates)].copy();dest=out/f'evaluation/{arm}_{seed}_{block}';dest.mkdir(parents=True);f.to_csv(dest/'predictions.csv',index=False)
                result=evaluate_frame(f,EvaluationConfig(test_start=dates[0],test_end=dates[-1],top_k=6,drop_n=1,cost_rate=.0015,min_holding_days=5,bootstrap_reps=1000,bootstrap_block=10))
                write_json(dest/'summary.json',result)
                for signal in ('path','endpoint'):
                    m=result['metrics'][signal];cross=pd.DataFrame(result['daily'][signal]['cross_section']);port=pd.DataFrame(result['daily'][signal]['portfolio']);dd=cross.merge(port,on='as_of_date',suffixes=('','_port'))
                    err=f[PCOLS].to_numpy()-f[ACOLS].to_numpy();daily_error=[]
                    for d,g in f.groupby('as_of_date'):
                        ee=g[PCOLS].to_numpy()-g[ACOLS].to_numpy();daily_error.append({'as_of_date':d,'trajectory_mae':float(abs(ee).mean()),'trajectory_mse':float((ee**2).mean())})
                    dd=dd.merge(pd.DataFrame(daily_error),on='as_of_date');dd.to_csv(dest/f'{signal}_daily.csv',index=False)
                    evaluations.append({'arm':arm,'seed':seed,'block':block,'signal':signal,'selected_epoch':best_epoch,'mae':float(abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),'direction_accuracy':m['direction']['accuracy'],'balanced_accuracy':m['direction']['balanced_accuracy'],'ic':m['cross_section']['ic']['mean'],'rankic':m['cross_section']['rankic']['mean'],'net_return':float(np.prod(1+dd.top_net)-1),'sharpe':m['portfolio']['sharpe'],'max_drawdown':m['portfolio']['max_drawdown'],'turnover':m['portfolio']['turnover']})
            pd.DataFrame(evaluations).to_csv(out/'metrics.csv',index=False);pd.DataFrame(selections).to_csv(out/'selection.csv',index=False)
            write_json(out/'progress.json',{'stage':'train','finished_arm':arm,'seed':seed,'seconds':time.perf_counter()-started})
            del model,opt
    write_json(out/'audit.json',{'frozen_checks':frozen_checks,'gradient_checks':gradient_checks,'test_selection':False,'initial_identity_atol':1e-6,'origin_date_batches':True})
    write_json(out/'status.json',{'status':'complete','seconds':time.perf_counter()-started,'rows':rows.groupby('split').size().to_dict(),'epochs':args.epochs,'test_dates':160,'seeds':[100,200,300]})
    manifest={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/kronos_continual/lora_suffix.py',ROOT/'evaluate/metrics.py']}
    write_json(out/'source_manifest.json',manifest)
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--epochs',type=int,default=3);args=parser.parse_args()
    try:run(args)
    except Exception as e:
        if args.output_dir.exists():write_json(args.output_dir/'FAILED.json',{'error':repr(e)})
        raise
