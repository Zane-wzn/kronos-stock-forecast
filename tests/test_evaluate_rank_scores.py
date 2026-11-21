import numpy as np
import pandas as pd
import pytest
from scripts.evaluation.evaluate_rank_scores import evaluate_scores,verify_score_replay


def fixture():
    rows=[];score=[];ys=[]
    for t,date in enumerate(pd.date_range('2025-01-01',periods=8).strftime('%Y-%m-%d')):
        stocks=['A','B','C','D'] if t<7 else ['A','B','C','E']
        for i,s in enumerate(stocks):
            rows.append(dict(as_of_date=date,stock=s))
            score.append(([4,3,2,1] if t==0 else [1,2,3,4])[i])
            ys.append(np.linspace((i+1)*.001,(i+1)*.01,10))
    return pd.DataFrame(rows),np.array(score),np.array(ys),dict(top_k=2,drop_n=1)


def test_hand_hold_lock_rebalance_and_double_costs():
    meta,s,y,cfg=fixture();ms,dd,hh=evaluate_scores(meta,s,y,cfg)
    for t in range(5):
        assert set(hh[(hh.as_of_date==dd.as_of_date.iloc[t])&hh.weight.gt(0)].stock)=={'A','B'}
    assert set(hh[(hh.as_of_date==dd.as_of_date.iloc[5])&hh.weight.gt(0)].stock)=={'B','D'}
    assert dd.transaction_cost.iloc[0]==pytest.approx(.0015)
    assert dd.transaction_cost.iloc[1:5].sum()==0
    assert dd.transaction_cost.iloc[5]==pytest.approx(.0015) # .5 buy+.5 sell
    assert dd.forced_exits.iloc[7]==1
    audit=verify_score_replay(meta,s,y,cfg,ms,dd,hh)
    assert audit['status']=='passed'


def test_input_permutation_ties_and_score_scale_invariance():
    meta,s,y,cfg=fixture();s[0:4]=1
    ms,dd,hh=evaluate_scores(meta,s,y,cfg)
    assert set(hh[hh.as_of_date.eq(dd.as_of_date.iloc[0])&hh.weight.gt(0)].stock)=={'A','B'}
    order=np.random.default_rng(2).permutation(len(meta))
    mm,ddd,hhh=evaluate_scores(meta.iloc[order],s[order]*7+20,y[order],cfg)
    np.testing.assert_allclose(dd.net_return,ddd.net_return)
    np.testing.assert_allclose(dd.topk_excess,ddd.topk_excess)
    pd.testing.assert_frame_equal(hh[['as_of_date','stock','weight']],hhh[['as_of_date','stock','weight']])


def test_no_score_rmse_and_replay_catches_cost_tampering():
    meta,s,y,cfg=fixture();ms,dd,hh=evaluate_scores(meta,s,y,cfg)
    assert not any('rmse' in key for key in ms)
    dd.loc[0,'transaction_cost']*=2
    with pytest.raises(AssertionError):verify_score_replay(meta,s,y,cfg,ms,dd,hh)


def test_alignment_rejects_duplicate_stock_or_wrong_shapes():
    meta,s,y,cfg=fixture();meta.loc[1,'stock']='A'
    with pytest.raises(ValueError):evaluate_scores(meta,s,y,cfg)
    meta,s,y,cfg=fixture()
    with pytest.raises(ValueError):evaluate_scores(meta,s[:-1],y,cfg)
