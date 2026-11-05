"""Causality checks for the actual expanded temporal pilot engine."""
from types import SimpleNamespace
import copy
import json
import numpy as np
import pandas as pd
import torch

from scripts.experiments.run_temporal_attention_pilot import (
    ROOT, ARCHITECTURES, load_model, online_predictions, select_panel,
)
from scripts.experiments.run_delayed_feedback_pilot import Model


def test_metadata_selection_and_split_boundaries():
    records=[]
    for split,start in [('train','2022-01-01'),('val','2023-01-01'),('test','2024-01-01')]:
        for d in pd.date_range(start,periods=40):
            for s in ('a','b','c','d'):
                records.append({'split':split,'as_of_date':str(d.date()),'stock':s,
                                'label_end_date':str((d+pd.Timedelta(days=10)).date())})
    m=pd.DataFrame(records)
    rows,cohort=select_panel(m,3,8,6,24)
    assert len(cohort)==3 and rows[rows.split.eq('test')].as_of_date.nunique()==24
    test_dates=sorted(rows[rows.split.eq('test')].as_of_date.unique())
    assert test_dates==sorted(m[m.split.eq('test')].as_of_date.unique())[:24]
    # Adding a ticker visible only in the test period cannot change the cohort.
    more=pd.concat([m,m[m.split.eq('test')].assign(stock='test_only')])
    _,again=select_panel(more,3,8,6,24)
    assert cohort==again


def test_actual_online_engine_maturity_and_future_poison(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str,
                        usecols=['as_of_date','label_dates']).drop_duplicates('as_of_date')
    dates=sorted(windows[windows.as_of_date.ge('2025-07-01')].as_of_date.unique())[:24]
    records=[{'split':'test','as_of_date':d,'stock':s,'row_id':i*4+j,
              'label_start_date':windows.set_index('as_of_date').loc[d,'label_dates'].split('|')[0],
              'label_end_date':windows.set_index('as_of_date').loc[d,'label_dates'].split('|')[-1]}
             for i,d in enumerate(dates) for j,s in enumerate(['a','b','c','d'])]
    rows=pd.DataFrame(records)
    h=np.random.default_rng(1).normal(size=(len(rows),90,512)).astype('float32')
    y=np.random.default_rng(2).normal(0,.01,size=(len(rows),10)).astype('float32')
    state=Model().eval().state_dict()
    clean={(a,100):load_model(a,state,100) for a in ARCHITECTURES}
    poisoned=copy.deepcopy(clean)
    args=SimpleNamespace(seeds=[100])
    outputs=[]
    poison=y.copy()
    poison[rows.label_end_date.gt(dates[-1])]=np.nan
    for name,models,values in [('clean',clean,y),('poison',poisoned,poison)]:
        dest=tmp_path/name;dest.mkdir();(dest/'checkpoints').mkdir()
        outputs.append(online_predictions(models,rows,h,values,dest,args))
        access=pd.read_csv(dest/'label_access_log.csv')
        assert (access.complete_date<=access.update_date).all()
        assert (access.label_date<=access.update_date).all()
        for _,g in access.groupby(['architecture','seed']):
            assert not g.duplicated(['row_id','horizon']).any()
        issued=next((dest/'issued/adapter_100').glob('*.csv'))
        assert not any(c.startswith('actual') for c in pd.read_csv(issued).columns)
    for key in clean:
        for k,v in clean[key].state_dict().items():
            torch.testing.assert_close(v,poisoned[key].state_dict()[k],rtol=0,atol=0)
        pc=[f'pred_return_{i}' for i in range(1,11)]
        np.testing.assert_array_equal(outputs[0][(*key,'online')][pc],outputs[1][(*key,'online')][pc])
