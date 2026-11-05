"""Re-evaluate saved Prediction stages on exactly the same panel; no training."""
import sys, json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from evaluate.inputs import normalize_predictions
from evaluate.metrics import EvaluationConfig, evaluate_frame, clean_json
from src.doubleadapt.compare import return_stats, indices

DATA=ROOT/'experiments/csi300_aligned'
OUT=DATA/'evaluations/prediction_progression_20261102'
KEYS=['as_of_date','stock']
PC=[f'pred_return_{k}' for k in range(1,11)]
AC=[f'actual_return_{k}' for k in range(1,11)]

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    ref=normalize_predictions(pd.read_csv(DATA/'models/prediction_adapter/predictions.csv',dtype={'stock':str})).sort_values(KEYS).reset_index(drop=True)
    paths={
        'Kronos_zero_shot':sorted((DATA/'models/zero_shot/predictions/by_date').glob('*.csv')),
        'Linear_head':[DATA/'models/linear_head/predictions.csv'],
        'Prediction_static':[DATA/'models/prediction_adapter/predictions.csv'],
        'Prediction_online':[DATA/'models/hca_gain_attribution_20261028/Prediction_Live_A/predictions.csv'],
        'Prediction_PCA_HCA':[DATA/'models/hca_gain_attribution_20261028/Prediction_Live_HCA/predictions.csv'],
    }
    cfg=EvaluationConfig(top_k=30,drop_n=3,min_holding_days=5,cost_rate=.0015,annualization_days=252,bootstrap_reps=0)
    rows=[];audits=[];daily={};points={};dates=sorted(ref.as_of_date.unique())
    for name,files in paths.items():
        raw=normalize_predictions(pd.concat([pd.read_csv(p,dtype={'stock':str}) for p in files],ignore_index=True))
        assert not raw.duplicated(KEYS).any()
        # Zero-shot uses a slightly larger universe; restrict to original locked panel.
        f=ref[KEYS].merge(raw,on=KEYS,how='left',validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        assert len(f)==66103 and f[KEYS].equals(ref[KEYS]) and np.isfinite(f[PC].to_numpy()).all()
        error=float(abs(f[AC].to_numpy()-ref[AC].to_numpy()).max())
        assert error<1e-6,(name,error)
        for c in AC: f[c]=ref[c].to_numpy()
        for prefix,cols in [('pred',PC),('actual',AC)]:
            f[prefix+'_signal']=f[cols].mean(axis=1)
            f[prefix+'_endpoint_return']=f[cols[-1]]
        result=evaluate_frame(f,cfg)
        (OUT/(name+'_evaluation.json')).write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        for mode in ['path','endpoint']:
            m=result['metrics'][mode];port=pd.DataFrame(result['daily'][mode]['portfolio'])
            cross=pd.DataFrame(result['daily'][mode]['cross_section'])
            daily[name,mode]=port.merge(cross,on='as_of_date',validate='one_to_one')
            daily[name,mode].to_csv(OUT/(name+'_'+mode+'_daily.csv'),index=False)
            rows.append(dict(stage=name,signal=mode,rows=len(f),trajectory_mae=result['metrics']['trajectory_pooled']['mae'],trajectory_rmse=result['metrics']['trajectory_pooled']['rmse'],
                mae=m['point']['mae'],rmse=m['point']['rmse'],direction_accuracy=m['direction']['accuracy'],balanced_accuracy=m['direction']['balanced_accuracy'],
                ic=m['cross_section']['ic']['mean'],rankic=m['cross_section']['rankic']['mean'],turnover=m['portfolio']['turnover'],**{k:float(v) for k,v in return_stats(port.top_net.to_numpy()).items()}))
            cols=PC if mode=='path' else [PC[-1]]
            pred=f[cols].mean(1).to_numpy();actual=f[AC].mean(1).to_numpy() if mode=='path' else f[AC[-1]].to_numpy()
            pp=pd.DataFrame({'date':f.as_of_date,'se':(pred-actual)**2,'correct':(pred>0)==(actual>0)})
            points[name,mode]=pp.groupby('date')[['se','correct']].sum().assign(n=pp.groupby('date').size())
        audits.append(dict(stage=name,source_rows=len(raw),matched_rows=len(f),target_max_error_before_canonicalization=error,sources=[str(p) for p in files]))
        print('evaluated '+name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    draws=indices(226,5000,10,20261013);pairs=[]
    for a,b in [('Prediction_static','Kronos_zero_shot'),('Prediction_static','Linear_head'),('Prediction_online','Prediction_static'),('Prediction_PCA_HCA','Prediction_online')]:
        for mode in ['path','endpoint']:
            la,lb=daily[a,mode],daily[b,mode]
            av,bv=la.top_net.to_numpy(),lb.top_net.to_numpy()
            dd=return_stats(av)['net_cumulative_return']-return_stats(bv)['net_cumulative_return']
            sample=return_stats(av[draws])['net_cumulative_return']-return_stats(bv[draws])['net_cumulative_return']
            lo,hi=np.quantile(sample,[.025,.975]);pairs.append(dict(left=a,right=b,signal=mode,metric='net_cumulative_return',delta=float(dd),ci_low=float(lo),ci_high=float(hi)))
            for metric in ['ic','rankic']:
                d=la[metric].to_numpy()-lb[metric].to_numpy();lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                pairs.append(dict(left=a,right=b,signal=mode,metric=metric,delta=float(d.mean()),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(pairs).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(status='passed',dates=226,rows=66103,common_targets=True,common_portfolio_config=True,
        caution='Retrospective fixed panel. Static-vs-linear may differ in objectives/checkpoint selection, not pure adapter intervention. No proof of anti-overfitting.',sources=audits),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(pairs).to_string(index=False),flush=True)

if __name__=='__main__':main()
