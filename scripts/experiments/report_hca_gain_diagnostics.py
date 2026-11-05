from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from scripts.experiments.run_hca_gain_attribution import old, PCOLS, KEYS
out=ROOT/'experiments/csi300_aligned/models/hca_gain_attribution_20261028'
t=pd.read_csv(out/'attribution_training_summary.csv').set_index('name'); rows=[]
for ck in ('Prediction','RankIC'):
 for mode in ('Live','Frozen'):
  prefix=f'{ck}_{mode}_'; a=t.loc[prefix+'A']; h=t.loc[prefix+'HCA']
  hp=pd.read_csv(out/(prefix+'HCA')/'causal_predictions.csv',dtype={'stock':str}).sort_values(KEYS).reset_index(drop=True)
  for v in ('A','A_MatchedStep','A_ProjectedUpdate','HCA_FrozenHead'):
   p=pd.read_csv(out/(prefix+v)/'causal_predictions.csv',dtype={'stock':str}).sort_values(KEYS).reset_index(drop=True)
   assert p[KEYS].equals(hp[KEYS]); x=p[PCOLS].to_numpy(); y=hp[PCOLS].to_numpy()
   ranks=[]; overlaps=[]
   for date,ids in p.groupby('as_of_date').groups.items():
    xx=x[ids].mean(1); yy=y[ids].mean(1)
    ranks.append(pd.Series(xx).rank().corr(pd.Series(yy).rank()))
    overlaps.append(len(set(np.argsort(xx)[-30:]) & set(np.argsort(yy)[-30:]))/30)
   rows.append(dict(checkpoint=ck,adapter=mode,comparison='HCA_vs_'+v,
    prediction_rmse=float(np.sqrt(((x-y)**2).mean())),mean_daily_signal_rank_correlation=float(np.mean(ranks)),
    mean_daily_top30_overlap=float(np.mean(overlaps)),hca_to_A_head_norm_ratio=h.effective_head_update_norm_mean/a.effective_head_update_norm_mean,
    hca_to_A_in_basis_ratio=h.update_basis_in_norm_mean/a.update_basis_in_norm_mean,
    hca_to_A_out_basis_ratio=h.update_basis_out_norm_mean/a.update_basis_out_norm_mean))
old.atomic_csv(out/'mechanism_prediction_diagnostics.csv',pd.DataFrame(rows)); print(pd.DataFrame(rows).to_string(index=False))
