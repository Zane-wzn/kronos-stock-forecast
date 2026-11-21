"""Independent scalar-score evaluation, separate from return prediction APIs."""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlation(x,y):
    if len(x)<2 or np.std(x)==0 or np.std(y)==0:return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def validate(meta,scores,actual):
    meta=meta.reset_index(drop=True).copy();meta['stock']=meta.stock.astype(str)
    scores=np.asarray(scores,dtype=float);actual=np.asarray(actual,dtype=float)
    if scores.shape!=(len(meta),) or actual.shape!=(len(meta),10):raise ValueError('unaligned scores/labels')
    if meta.duplicated(['as_of_date','stock']).any():raise ValueError('duplicate daily stock IDs')
    if not np.isfinite(scores).all() or not np.isfinite(actual).all():raise ValueError('nonfinite inputs')
    return meta,scores,actual


def portfolio_statistics(net):
    net=np.asarray(net,float);wealth=np.cumprod(1+net)
    peaks=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    sd=net.std(ddof=1) if len(net)>1 else 0
    return dict(net_cumulative_return=float(wealth[-1]-1),sharpe=float(net.mean()/sd*np.sqrt(252)) if sd else float('nan'),
        max_drawdown=float((wealth/peaks-1).min()))


def evaluate_scores(meta,scores,actual,cfg,base_scores=None):
    meta,scores,actual=validate(meta,scores,actual)
    base=scores if base_scores is None else np.asarray(base_scores,float)
    if base.shape!=scores.shape or not np.isfinite(base).all():raise ValueError('unaligned base scores')
    previous={};opened={};rows=[];holdings=[];minhold=cfg.get('min_hold',5)
    for t,(date,g) in enumerate(meta.groupby('as_of_date',sort=True)):
        ids=g.index.to_numpy();stocks=g.stock.to_numpy();s=scores[ids];y=actual[ids,9];r=actual[ids,0]
        order=np.lexsort((stocks,-s));ranking=list(stocks[order]);k=min(cfg['top_k'],len(ids))
        desired=set(ranking[:k]);universe=set(stocks);lookup={stock:i for i,stock in enumerate(stocks)}
        forced=set(previous)-universe
        eligible=[stock for stock in previous if stock in universe and stock not in desired and t-opened[stock]>=minhold]
        eligible.sort(key=lambda stock:(s[lookup[stock]],stock))
        sell=forced|set(eligible[:cfg['drop_n']]);held=set(previous)-sell
        if not previous:held=set(ranking[:k])
        elif len(held)<k:
            for stock in ranking[:k]:
                if len(held)>=k:break
                held.add(stock)
        if len(held)>k:held=set(sorted(held,key=lambda stock:(-s[lookup[stock]],stock))[:k])
        weights={stock:1/len(held) for stock in held}
        names=set(previous)|held
        buy_notional=sum(max(weights.get(stock,0)-previous.get(stock,0),0) for stock in names)
        sell_notional=sum(max(previous.get(stock,0)-weights.get(stock,0),0) for stock in names)
        turnover=1. if not previous else .5*(buy_notional+sell_notional)
        cost=.0015*(buy_notional+sell_notional)
        gross=float(np.mean([r[lookup[stock]] for stock in held]))
        bought=held-set(previous);sold=set(previous)-held
        comparable_sold=[stock for stock in sold if stock in universe]
        swap=float(np.mean([y[lookup[stock]] for stock in bought])-np.mean([y[lookup[stock]] for stock in comparable_sold])) if bought and comparable_sold else float('nan')
        true_top=set(stocks[np.lexsort((stocks,-y))[:k]])
        ranks=rankdata(-base[ids],method='average')
        zone=(ranks>=math.ceil(.5*k))&(ranks<=2*k)
        ii,jj=np.triu_indices(len(ids),1);valid=(zone[ii]|zone[jj])&(y[ii]!=y[jj])
        products=(s[ii[valid]]-s[jj[valid]])*(y[ii[valid]]-y[jj[valid]])
        accuracy=float(np.mean((products>0)+.5*(products==0))) if len(products) else float('nan')
        rows.append(dict(as_of_date=str(date),rankic=correlation(rankdata(s),rankdata(y)),ic=correlation(s,y),
            topk_excess=float(y[order[:k]].mean()-y.mean()),topk_hit=len(desired&true_top)/k,
            boundary_accuracy=accuracy,net_return=gross-cost,gross_return=gross,transaction_cost=cost,turnover=turnover,
            benchmark_daily_return=float(r.mean()),swap_excess=swap,bought=len(bought),sold=len(sold),
            forced_exits=len(forced),buy_notional=buy_notional,sell_notional=sell_notional))
        for stock in sorted(names):
            pos=lookup.get(stock)
            holdings.append(dict(as_of_date=str(date),stock=stock,weight=weights.get(stock,0),previous_weight=previous.get(stock,0),
                age=t-opened.get(stock,t),opened_index=opened.get(stock,t),bought=stock in bought,sold=stock in sold,
                forced_exit=stock in forced,score=float(s[pos]) if pos is not None else float('nan'),
                actual_return_1=float(r[pos]) if pos is not None else float('nan'),endpoint_return=float(y[pos]) if pos is not None else float('nan'),
                buy_notional=max(weights.get(stock,0)-previous.get(stock,0),0),sell_notional=max(previous.get(stock,0)-weights.get(stock,0),0)))
        opened={stock:opened.get(stock,t) for stock in held};previous=weights
    daily=pd.DataFrame(rows);hh=pd.DataFrame(holdings)
    metrics={col:float(daily[col].mean()) for col in ('rankic','ic','topk_excess','topk_hit','boundary_accuracy','turnover','swap_excess')}
    metrics.update(portfolio_statistics(daily.net_return))
    sd=daily.rankic.std(ddof=1)
    metrics.update(rankic_ir=float(daily.rankic.mean()/sd) if sd>0 else float('nan'),dates=len(daily),rows=len(meta),
        mean_transaction_cost=float(daily.transaction_cost.mean()))
    return metrics,daily,hh


def replay_portfolio(meta,scores,actual_day,cfg):
    """Rebuild TopK/DropN holdings without calling the evaluation implementation."""
    previous=set();opened={};gross=[];turnovers=[]
    for t,(_,g) in enumerate(meta.groupby('as_of_date',sort=True)):
        ids=g.index.to_numpy();stocks=g.stock.to_numpy();value=dict(zip(stocks,scores[ids]))
        ranking=sorted(stocks,key=lambda s:(-value[s],s));k=min(cfg['top_k'],len(stocks))
        desired=set(ranking[:k]);universe=set(stocks)
        missing=previous-universe
        removable=[s for s in previous&universe if s not in desired and t-opened[s]>=5]
        removable.sort(key=lambda s:(value[s],s))
        held=previous-missing-set(removable[:cfg['drop_n']])
        if not previous:held=set(ranking[:k])
        for s in ranking[:k]:
            if len(held)>=k:break
            held.add(s)
        if len(held)>k:held=set(sorted(held,key=lambda s:(-value[s],s))[:k])
        oldweight=1/len(previous) if previous else 0;weight=1/len(held)
        turnover=1. if not previous else .5*sum(abs((weight if s in held else 0)-(oldweight if s in previous else 0)) for s in held|previous)
        gross.append(float(np.mean([r for s,r in zip(stocks,actual_day[ids]) if s in held])))
        turnovers.append(turnover)
        opened={s:opened.get(s,t) for s in held};previous=held
    return np.asarray(gross),np.asarray(turnovers)


def verify_score_replay(meta,scores,actual,cfg,metrics,daily,holdings,base_scores=None):
    """Use a separate state machine and reconstruct costs from emitted weights."""
    meta,scores,actual=validate(meta,scores,actual)
    gross,turnover=replay_portfolio(meta,scores,actual[:,0],cfg)
    daily=daily.sort_values('as_of_date').reset_index(drop=True)
    np.testing.assert_allclose(daily.gross_return,gross,rtol=0,atol=1e-12)
    np.testing.assert_allclose(daily.turnover,turnover,rtol=0,atol=1e-12)
    costs=[];held_gross=[];prior={}
    for date,g in holdings.groupby('as_of_date',sort=True):
        if g.stock.duplicated().any():raise AssertionError('duplicate holding')
        weights=dict(zip(g.stock.astype(str),g.weight))
        np.testing.assert_allclose(sum(weights.values()),1,rtol=0,atol=1e-12)
        # Explicit previous-weight entries ensure omitted sales cannot hide costs.
        expected=set(prior)|{s for s,w in weights.items() if w>0}
        assert expected==set(weights)
        for r in g.itertuples():
            assert abs(r.previous_weight-prior.get(str(r.stock),0))<1e-12
        traded=sum(abs(weights.get(s,0)-prior.get(s,0)) for s in set(weights)|set(prior))
        costs.append(.0015*traded)
        selected=g[g.weight.gt(0)]
        held_gross.append(float((selected.weight*selected.actual_return_1).sum()))
        prior={s:w for s,w in weights.items() if w>0}
    np.testing.assert_allclose(held_gross,gross,rtol=0,atol=1e-12)
    np.testing.assert_allclose(daily.transaction_cost,costs,rtol=0,atol=1e-12)
    net=gross-np.asarray(costs)
    np.testing.assert_allclose(daily.net_return,net,rtol=0,atol=1e-12)
    for key,value in portfolio_statistics(net).items():np.testing.assert_allclose(metrics[key],value,rtol=0,atol=1e-12,equal_nan=True)
    ranks=[];topk=[]
    for date,g in meta.groupby('as_of_date',sort=True):
        ids=g.index.to_numpy();s=pd.Series(scores[ids]);y=pd.Series(actual[ids,9])
        ranks.append(s.rank(method='average').corr(y.rank(method='average')))
        order=sorted(range(len(ids)),key=lambda i:(-scores[ids[i]],str(g.stock.iloc[i])))[:cfg['top_k']]
        topk.append(float(actual[ids[order],9].mean()-actual[ids,9].mean()))
    np.testing.assert_allclose(daily.rankic,ranks,rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(daily.topk_excess,topk,rtol=0,atol=1e-12)
    np.testing.assert_allclose(metrics['rankic'],np.nanmean(ranks),rtol=0,atol=1e-12)
    return dict(status='passed',dates=len(daily),independent_portfolio_state_machine=True,
        holdings_costs_and_net=True,daily_rankic_and_topk=True)



def evaluate_signal(meta,scores,actual,cfg,signal,base_scores=None):
    """Evaluate a dimensionless score against the signal's own return target."""
    if signal not in ('path','endpoint'):raise ValueError(f'unknown signal {signal}')
    target=np.array(actual,dtype=np.float64,copy=True)
    if signal=='path':target[:,9]=np.asarray(actual,dtype=np.float64).mean(1)
    metrics,daily,holdings=evaluate_scores(meta,scores,target,cfg,base_scores)
    audit=verify_score_replay(meta,scores,target,cfg,metrics,daily,holdings,base_scores)
    metrics['net_annualized_return']=(1+metrics['net_cumulative_return'])**(252/len(daily))-1
    return metrics,daily,holdings,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta',type=Path,required=True);parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True);parser.add_argument('--method',default='R1')
    parser.add_argument('--top-k',type=int,required=True);parser.add_argument('--drop-n',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();meta=pd.read_csv(args.meta,dtype={'stock':str});raw=np.load(args.scores)
    y=np.load(args.labels);cfg=dict(top_k=args.top_k,drop_n=args.drop_n)
    metrics,daily,holdings=evaluate_scores(meta,raw[args.method],y,cfg,raw['B0'])
    audit=verify_score_replay(meta,raw[args.method],y,cfg,metrics,daily,holdings,raw['B0'])
    args.output.mkdir(parents=True,exist_ok=False)
    daily.to_csv(args.output/'daily_metrics.csv',index=False);holdings.to_csv(args.output/'holdings.csv.gz',index=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (args.output/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')


if __name__=='__main__':main()
