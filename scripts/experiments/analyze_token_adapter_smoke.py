"""Analyze fixed saved token-adapter smoke; no training or model selection."""
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.runtime/torchinductor_cache'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,pandas as pd,torch
from src.kronos_continual.token_adapter import TokenPredictionAdapter
from scripts.experiments.run_token_adapter_smoke import token_metrics
from scripts.experiments.run_temporal_attention_pilot import PCOLS,ACOLS,write_json


def main():
    out=ROOT/'experiments/csi300_aligned/models/token_adapter_smoke_20261105'
    dest=out/'analysis';dest.mkdir(exist_ok=True)
    f=pd.read_csv(out/'predictions.csv',dtype={'stock':str}); keys=['as_of_date','stock']
    reference=f[f.method.eq('official_zero_shot')].sort_values(keys).reset_index(drop=True)
    old=pd.read_csv(ROOT/'experiments/csi300_aligned/models/prediction_adapter/predictions.csv',dtype={'stock':str})
    old=reference[keys].merge(old,on=keys,validate='one_to_one').sort_values(keys).reset_index(drop=True)
    assert len(old)==16
    delta=float(abs(old[ACOLS].to_numpy()-reference[ACOLS].to_numpy()).max());assert delta<1e-6
    old[ACOLS]=reference[ACOLS].to_numpy();old['method']='original_return_adapter'
    f=pd.concat([f,old[f.columns.intersection(old.columns)]],ignore_index=True)
    summaries=[];daily=[];horizons=[]
    for name,g in f.groupby('method'):
        g=g.sort_values(keys);p=g[PCOLS].to_numpy();y=g[ACOLS].to_numpy();e=p-y
        rec={'method':name,'mae':float(abs(e).mean()),'rmse':float(np.sqrt((e**2).mean())),'bias':float(e.mean()),'trajectory_direction':float(((p>0)==(y>0)).mean())}
        for mode,pp,yy in [('path',p.mean(1),y.mean(1)),('endpoint',p[:,-1],y[:,-1])]:
            dd=pd.DataFrame({'date':g.as_of_date.to_numpy(),'pred':pp,'actual':yy});cs=[]
            for date,part in dd.groupby('date'):
                ic=part.pred.corr(part.actual);ric=part.pred.corr(part.actual,method='spearman');cs.append((ic,ric))
                daily.append({'method':name,'signal':mode,'date':date,'mae':float(abs(part.pred-part.actual).mean()),'ic':ic,'rankic':ric})
            rec[mode+'_ic']=float(np.mean([x[0] for x in cs]));rec[mode+'_rankic']=float(np.mean([x[1] for x in cs]));rec[mode+'_direction']=float(((pp>0)==(yy>0)).mean())
        summaries.append(rec)
        for k in range(10):horizons.append({'method':name,'horizon':k+1,'mae':float(abs(e[:,k]).mean()),'rmse':float(np.sqrt((e[:,k]**2).mean())),'bias':float(e[:,k].mean())})
    pd.DataFrame(summaries).to_csv(dest/'prediction_metrics.csv',index=False);pd.DataFrame(daily).to_csv(dest/'daily_metrics.csv',index=False);pd.DataFrame(horizons).to_csv(dest/'horizon_metrics.csv',index=False)
    torch.set_num_threads(4);sys.path.insert(0,str(ROOT/'third_party/Kronos_official_src/Kronos-master'))
    from model import Kronos
    hub=ROOT/'.runtime/hf/hub/models--NeoQuasar--Kronos-small/snapshots';md=sorted(hub.iterdir())[0]
    base=Kronos.from_pretrained(str(md)).eval()
    cache=torch.load(out/'tokens.pt',weights_only=True);tokens=(cache['coarse'],cache['fine']);ts=cache['timestamps']
    rows=pd.read_csv(out/'selected_samples.csv');ids=np.flatnonzero(rows.split.eq('test'))
    tm=[]
    for mode in ('identity','return_checkpoint'):
        model=TokenPredictionAdapter(base).eval()
        if mode=='return_checkpoint':model.load_return_adapter(ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt')
        tm.append({'mode':mode,'stage':'untrained_test',**token_metrics(model,tokens,ts,ids)})
        state=torch.load(out/f'checkpoints/{mode}.pt',weights_only=True);model.adapter.load_state_dict(state['adapter'])
        tm.append({'mode':mode,'stage':'selected_test',**token_metrics(model,tokens,ts,ids)})
    pd.DataFrame(tm).to_csv(dest/'paired_test_token_metrics.csv',index=False)
    write_json(dest/'alignment.json',{'test_rows':16,'dates':4,'stocks':4,'original_return_label_max_error':delta,'no_training':True,'no_portfolio_statistics':'only4 nonconsecutive dates; insufficient evidence'})
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(tm).to_string(index=False));print(pd.DataFrame(horizons).pivot(index='horizon',columns='method',values='mae').to_string())

if __name__=='__main__':main()
