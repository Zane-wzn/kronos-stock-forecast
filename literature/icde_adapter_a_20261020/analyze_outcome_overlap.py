"""Read-only retrospective target-identity diagnostic, not an online training run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
cache = ROOT / 'experiments/csi300_aligned/doubleadapt_cache_v2'
meta = pd.read_csv(cache/'meta.csv', dtype={'stock':str})
test = meta[meta.split.eq('test')].copy()
windows = pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',
                      usecols=['as_of_date','label_dates'], dtype=str)
windows = windows.drop_duplicates('as_of_date').set_index('as_of_date')
y = np.asarray(np.load(cache/'y.npy', mmap_mode='r')[test.row_id.to_numpy()], dtype=np.float64)
assert np.all(y > -1)
q = np.diff(np.concatenate([np.zeros((len(y),1)),np.log1p(y)],axis=1),axis=1)
parts=[]
for k in range(10):
    frame=test[['stock','as_of_date']].copy()
    frame['realization_date']=frame.as_of_date.map(windows.label_dates.str.split('|').str[k])
    frame['implied_log_daily_return']=q[:,k]
    parts.append(frame)
flat=pd.concat(parts,ignore_index=True)
g=flat.groupby(['stock','realization_date']).implied_log_daily_return.agg(['size','min','max'])
result={
    'purpose':'retrospective target identity only; not adaptation performance evidence',
    'origin_stock_rows':len(test), 'origin_stock_horizon_constraints':len(flat),
    'unique_stock_realization_dates':len(g),
    'mean_constraints_per_outcome':float(g['size'].mean()),
    'max_constraints_per_outcome':int(g['size'].max()),
    'max_cross_origin_log_daily_return_disagreement':float((g['max']-g['min']).max()),
    'note':'Overlapping constraints are correlated; they are not interchangeable independent labels. Float32 cumulative returns introduce rounding.',
}
out=Path(__file__).with_name('outcome_overlap.json')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
