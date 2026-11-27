"""Replay three paper policies with adjusted supervision and corporate-action accounting."""
import argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch

from .prepare import HERE, ROOT, KEYS, load_panel, label_prices
from scripts.finca.next_open_main_20261123.replay_markets import LIMITS, load_quotes
from scripts.finca.next_open_main_20261123.open_execution import run_open as original_open
from scripts.finca.csi300_adjusted_20261124.accounting import run_open
from scripts.evaluation.evaluate_rank_scores import evaluate_scores, portfolio_statistics
from scripts.evaluation.paired_statistics import block_indices
from scripts.experiments import peer_pca_replay as replay, peer_pca_correction as pca
from src.kronos_continual.online_peer_pca_head import cross_section_weights

MODELS = ('Static Adapter', 'Online Head Only', 'Full Model')


def peer_weights(close, panels):
    values = []
    for panel in panels:
        prices = label_prices(close, panel.as_of_date, panel.stock)
        values.append(panel[KEYS].assign(close=prices))
    prices = pd.concat(values).pivot(index='as_of_date',columns='stock',values='close').sort_index()
    returns = prices.pct_change(fill_method=None)
    returns.loc[[panels[1].as_of_date.min(), panels[2].as_of_date.min()]] = np.nan
    bounds = returns.loc[panels[0].as_of_date.min():panels[0].as_of_date.max()].quantile([.01,.99])
    returns = returns.clip(lower=bounds.loc[.01],upper=bounds.loc[.99],axis=1)
    weights = {}
    for date in sorted(panels[2].as_of_date.unique()):
        end = returns.index.get_loc(date)
        weights[date] = cross_section_weights(returns.iloc[max(0,end-60):end].to_numpy(),40)
    return dict(zip(returns.columns,range(len(returns.columns)))), weights


def forecast(out, seed, context, features, labels, meta, weights, schedule):
    checkpoint = out / f'seed{seed}/model.pt'
    groups = meta.groupby('as_of_date',sort=True).indices
    targets = torch.from_numpy(labels)
    for name in MODELS:
        model = replay.make_model(checkpoint, name != MODELS[0])
        params = [dict(params=model.head.parameters(),lr=1e-4)]
        if name == MODELS[2]:
            model.adapter.requires_grad_(True)
            params.append(dict(params=model.adapter.parameters(),lr=1e-5))
        optimizer = torch.optim.AdamW(params,weight_decay=0) if name != MODELS[0] else None
        observe, correction, counts, updates = pca.correction_observer(meta,*weights,context)
        scores = np.empty(len(meta))
        for date, ids in groups.items():
            with torch.no_grad():
                encoded = model.adapter(features[ids])
                scores[ids] = model.head(encoded).numpy()[:,-1]
                observe(date,ids,encoded,model.head)
            if optimizer is not None and date in schedule:
                ids = schedule[date]
                replay.daily_update(model,optimizer,features[ids],targets[ids],256)
        pca.verify_corrections(correction,counts)
        assert len(updates)==len(groups)
        yield name, scores + .025 * correction[:,-1]


def paired(daily):
    rows = []
    for scenario, frame in daily.groupby('scenario'):
        for metric in ('net_return', 'rankic'):
            aligned = frame.pivot(index='as_of_date',columns=['model','seed'],values=metric).sort_index()
            if any(tuple(aligned[name].columns) != replay.SEEDS for name in MODELS):
                raise ValueError('missing or unaligned seeds')
            def statistic(values, axis):
                if metric=='net_return':
                    return np.prod(1+values,axis=axis)**(252/len(aligned))-1
                return values.mean(axis=axis)
            joint=aligned[MODELS[-1]].to_numpy()
            for name in MODELS[:2]:
                control=aligned[name].to_numpy()
                point=(statistic(joint,0)-statistic(control,0)).mean()
                for block in (10,20,40):
                    indices=block_indices(len(aligned),block)
                    distribution=(statistic(joint[indices],1)-statistic(control[indices],1)).mean(axis=1)
                    low,high=np.percentile(distribution,[2.5,97.5])
                    rows.append(dict(scenario=scenario,metric=metric,reference=name,block_days=block,delta=point,low=low,high=high))
    return pd.DataFrame(rows)


def main(market):
    out = HERE / market
    archive = HERE.parent / 'next_open_main_20261123' / market
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    original, full, calendar = load_panel(market)
    old_opens, old_closes = load_quotes(market,full,calendar)
    matrices = [pd.read_csv(out / f'{name}.csv',index_col=0) for name in ('opens','closes','dividends','splits')]
    assert all(m.index.equals(old_opens.index) and m.columns.equals(old_opens.columns) for m in matrices)
    original_test = pd.read_csv(cache/'meta_test.csv',usecols=KEYS+['label_end_date'])
    train,val = [pd.read_csv(cache/f'meta_{s}.csv',usecols=KEYS) for s in ('train','val')]
    hidden = np.load(cache/'hidden_test.npy')
    labels = np.load(out/'labels_test.npy')
    schedule = {date: ids for date, ids in replay.make_schedule(original_test).items() if len(ids)}
    assert len(schedule)==225 and sum(map(len,schedule.values()))=={'sp500':112454,'nasdaq100':22050}[market]
    assert all(np.isfinite(labels[ids]).all() for ids in schedule.values())
    if market=='sp500':
        with np.load(archive.parent/'dasfaa_extra_features_sp500.npz') as extra:
            hidden = np.concatenate([hidden,extra['hidden']])
        labels = np.concatenate([labels,np.load(out/'labels_extra.npy')])
    assert len(hidden)==len(labels)==len(full)
    features = torch.from_numpy(hidden)
    old = pd.read_csv(archive/'open_daily.csv').query("pool == 'full' and model in @MODELS")
    old = old.assign(scenario='paper')
    metrics,daily=[],[old]
    top,drop,_,days=LIMITS[market]
    cfg=dict(top_k=top,drop_n=drop,min_hold=5)
    groups = full.groupby('as_of_date',sort=True).indices
    adj = pd.read_csv(out/'adjusted_close.csv.gz',index_col=0)
    weights = peer_weights(adj,[train,val,full])
    training_hidden = np.load(cache/'hidden_train.npy',mmap_mode='r')

    def record(scenario, seed, name, scores):
        _,_,holdings=evaluate_scores(full[KEYS],scores,np.zeros_like(labels),cfg)
        targets = (holdings.pivot(index='as_of_date',columns='stock',values='weight')
                   .reindex(index=calendar.index,columns=old_opens.columns).fillna(0.))
        assert targets.gt(0).sum(axis=1).eq(top).all()
        if scenario=='replay_only':
            raw=original_open(targets,calendar,old_opens,old_closes)
            expected=old.loc[(old.seed==seed)&(old.model==name)].set_index('as_of_date')
            np.testing.assert_allclose(raw.net_return,expected.loc[raw.as_of_date,'net_return'],atol=1e-12,rtol=0)
        replayed=run_open(targets,calendar,*matrices)
        ranks = {}
        for date, ids in groups.items():
            ids=ids[np.isfinite(labels[ids,-1])]
            ranks[date]=spearmanr(scores[ids],labels[ids,-1]).statistic
        replayed['rankic']=replayed.as_of_date.map(ranks)
        err=scores-labels[:,-1]
        direction = np.isfinite(err)
        row=dict(scenario=scenario,model=name,seed=seed,rankic=replayed.rankic.mean(),
                 rmse=np.sqrt(np.nanmean(err**2)),mae=np.nanmean(np.abs(err)),
                 direction_accuracy=np.mean((scores[direction]>0)==(labels[direction,-1]>0)),
                 dividend_total=replayed.cash_dividend.sum(),**portfolio_statistics(replayed.net_return))
        metrics.append(row)
        daily.append(replayed.assign(scenario=scenario,model=name,seed=seed))
        print(market,scenario,seed,name,(1+row['net_cumulative_return'])**(252/days)-1,row['rankic'],flush=True)

    for (name,seed), frame in old.groupby(['model','seed']):
        metrics.append(dict(scenario='paper',model=name,seed=seed,rankic=frame.rankic.mean(),
                            **portfolio_statistics(frame.net_return)))
    for seed in replay.SEEDS:
        for name in MODELS:
            file=archive/f'endpoint_{seed}_{name.replace(" ","_")}.csv.gz'
            aligned=full[KEYS].merge(pd.read_csv(file),on=KEYS,how='left',validate='one_to_one')
            assert np.isfinite(aligned.score).all()
            record('replay_only',seed,name,aligned.score.to_numpy())
        checkpoint=out/f'seed{seed}/model.pt'
        context=pca.fit_training_context(train,training_hidden,replay.make_model(checkpoint,False).adapter)
        for name,scores in forecast(out,seed,context,features,labels,full,weights,schedule):
            full[KEYS].assign(score=scores).to_csv(out/f'endpoint_{seed}_{name.replace(" ","_")}.csv.gz',index=False)
            record('adjusted',seed,name,scores)
    by_seed=pd.DataFrame(metrics)
    by_seed['annualized_return']=(1+by_seed.net_cumulative_return)**(252/days)-1
    by_seed.to_csv(out/'by_seed.csv',index=False)
    daily=pd.concat(daily,ignore_index=True)
    daily.to_csv(out/'daily.csv',index=False)
    summary=by_seed.drop(columns='seed').groupby(['scenario','model']).mean().reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    paired(daily).to_csv(out/'paired_intervals.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('market',choices=['nasdaq100','sp500'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main(args.market)
