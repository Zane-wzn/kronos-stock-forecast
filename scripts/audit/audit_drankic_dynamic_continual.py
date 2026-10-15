"""Audit the locked dynamic D-RankIC prequential outputs without training or tuning."""
from __future__ import annotations
import argparse
import ast
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_dynamic_continual_csi300 import _load_and_join, build_dynamic_predictions


def audit(static_dir, continual_dir, dynamic_dir):
    s = pd.read_csv(static_dir/'predictions.csv', dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
    c = pd.read_csv(continual_dir/'predictions.csv', dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
    d = pd.read_csv(dynamic_dir/'predictions.csv', dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
    key=['as_of_date','stock']
    for frame in (s,c,d):
        assert len(frame)==66103 and frame.as_of_date.nunique()==226 and not frame[key].duplicated().any()
        assert frame.as_of_date.min()=='2025-07-01' and frame.as_of_date.max()=='2026-06-05'
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
        assert frame[key].equals(s[key])
    labels=['actual_signal','actual_endpoint_return']+[f'actual_return_{i}' for i in range(1,11)]
    np.testing.assert_allclose(s[labels],c[labels],atol=1e-15,rtol=0)
    np.testing.assert_allclose(s.actual_signal,d.actual_signal,atol=1e-15,rtol=0)
    initialization=json.loads((continual_dir/'initial_pair_audit.json').read_text())
    assert initialization['status']=='passed' and initialization['max_abs_diff']<=1e-6
    config=json.loads((continual_dir/'run_config.json').read_text())
    assert config['hard_acceptance_gate'] is False and config['seed']==100
    assert config['recent_validation_days']==30 and config['minimum_recent_validation_days']==20
    updates=pd.read_csv(continual_dir/'update_log.csv')
    versions=pd.Series(0,index=c.index)
    dates=sorted(c.as_of_date.unique())
    trained=0
    audits=[]
    for row in updates.to_dict('records'):
        u=pd.Timestamp(row['update_date']); nxt=row['next_test_date']
        assert nxt==dates[dates.index(row['update_date'])+1]
        assert c.loc[c.as_of_date.eq(row['update_date']),'model_version'].eq(row['model_version_before']).all()
        if row['candidate_trained']:
            trained+=1
            gate=pd.read_csv(continual_dir/'gate_audit'/f'{u.date()}.csv',dtype={'stock':str},parse_dates=['as_of_date','label_start_date','label_end_date'])
            replay=pd.read_csv(continual_dir/'replay_audit'/f'{u.date()}.csv',dtype={'stock':str},parse_dates=['as_of_date','label_start_date','label_end_date'])
            assert 20<=gate.as_of_date.nunique()<=30
            assert (gate.as_of_date<u).all() and (gate.label_end_date<=u).all()
            past=c[c.as_of_date.lt(str(u.date()))].copy()
            complete=[date for date, group in past.groupby('as_of_date')
                      if pd.to_datetime(group.label_end_date).le(u).all()]
            expected_gate=past[past.as_of_date.isin(complete[-30:])][key].sort_values(key).reset_index(drop=True)
            gate_keys=gate[key].copy(); gate_keys['as_of_date']=gate_keys.as_of_date.dt.strftime('%Y-%m-%d')
            pd.testing.assert_frame_equal(gate_keys.sort_values(key).reset_index(drop=True),expected_gate)
            assert (replay.label_end_date<gate.label_start_date.min()).all()
            assert (replay.label_end_date<=u).all()
            assert not set(map(tuple,gate[key].to_numpy())) & set(map(tuple,replay[key].to_numpy()))
            assert not replay[['_source','_row_id']].duplicated().any()
            assert row['promotion_ran'] and row['promotion_epochs']==1
            assert row['promotion_head_lr']==3e-5 and row['promotion_adapter_lr']==1e-5
            assert pd.Timestamp(row['promotion_max_label_end_date'])<=u
            assert row['deployment_effective_date']==nxt
            assert row['model_version_after']==row['model_version_before']+1
            history=ast.literal_eval(row['history']) if isinstance(row['history'],str) else row['history']
            assert row['best_epoch']==max(history,key=lambda x:x['val_rankic'])['epoch']
            following=dates[dates.index(nxt):dates.index(nxt)+20]
            mask=c.as_of_date.isin(following)
            change=float(np.max(np.abs(c.loc[mask,'pred_signal'].to_numpy()-s.loc[mask,'pred_signal'].to_numpy())))
            audits.append({'update_date':str(u.date()),'next_test_date':nxt,'gate_dates':gate.as_of_date.nunique(),
                'replay_rows':len(replay),'following20_max_prediction_difference_from_static':change})
        else:
            assert row['recent_validation_dates']<20 and row['model_version_after']==row['model_version_before']
        versions.loc[c.as_of_date.ge(nxt)]=int(row['model_version_after'])
    np.testing.assert_array_equal(c.model_version.to_numpy(),versions.to_numpy())
    before_first=c.model_version.eq(0)
    np.testing.assert_allclose(c.loc[before_first,[f'pred_return_{i}' for i in range(1,11)]],
                               s.loc[before_first,[f'pred_return_{i}' for i in range(1,11)]],atol=1e-6,rtol=0)
    joined=_load_and_join(static_dir/'predictions.csv',continual_dir/'predictions.csv')
    reproduced,weights=build_dynamic_predictions(joined,initial_alpha=.3)
    np.testing.assert_allclose(d[['alpha','final_signal']],reproduced[['alpha','final_signal']],atol=1e-12,rtol=0)
    assert weights.matured_dates.le(60).all()
    log=pd.read_csv(dynamic_dir/'weight_log.csv')
    np.testing.assert_allclose(log.alpha_after,weights.alpha_after,atol=1e-12,rtol=0)
    for row in weights.itertuples():
        window=joined[(joined.as_of_date<row.update_date)&(joined.label_end_date<=row.update_date)]
        eligible_dates=sorted(window.as_of_date.unique())[-60:]
        assert row.matured_rows==int(window.as_of_date.isin(eligible_dates).sum())
        np.testing.assert_allclose(d.loc[d.as_of_date.eq(str(row.update_date.date())),'alpha'],row.alpha_before)
        np.testing.assert_allclose(d.loc[d.as_of_date.eq(str(row.next_test_date.date())),'alpha'],row.alpha_after)
    result={'status':'passed','rows':66103,'dates':226,'training_updates':trained,
            'skipped_updates':len(updates)-trained,'initialization':initialization,'updates':audits}
    out=dynamic_dir/'evaluation'
    out.mkdir(exist_ok=True)
    (out/'protocol_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--static-dir',type=Path,required=True)
    p.add_argument('--continual-dir',type=Path,required=True)
    p.add_argument('--dynamic-dir',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.static_dir,a.continual_dir,a.dynamic_dir),indent=2))
