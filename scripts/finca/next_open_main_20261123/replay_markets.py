"""Replay all reported portfolios at the next recorded open."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .open_execution import run_open
from scripts.evaluation.evaluate_rank_scores import evaluate_scores, portfolio_statistics
from scripts.evaluation.paired_statistics import block_indices

from ..paths import ARCHIVED_RESULTS, DATA, ROOT, RUNS, unique_path

RUN = RUNS / 'next_open_main_20261123'
OLD = RUNS / 'csi300_next_open_20261123'
KEYS = ['as_of_date', 'stock']
SEEDS = (100, 101, 102)
MODELS = ('Static Adapter', 'Online Head Only', 'Full Model')
RAW = {'csi300':'csi300_daily/csi300_daily',
       'nasdaq100':'nasdaq100_2016_20260819/daily',
       'sp500':'yahoo_sp500/yahoo_sp500'}
LIMITS = {'csi300':(30,3,66223,226), 'nasdaq100':(10,1,23030,235),
          'sp500':(50,5,117474,235)}
EXTERNAL = {'Kronos (zero-shot)':'Kronos_zero_shot', 'Chronos-2':'Chronos2',
            'TimesFM 3.0':'TimesFM3', 'DoubleAdapt (GRU)':'DoubleAdapt_GRU',
            'SSPT (ind-SCC)':'SSPT_ind_SCC', 'MASTER (OHLCV)':'MASTER_OHLCV'}


def read_csv(path, cols=None):
    return pd.read_csv(path, usecols=cols, dtype={key:str for key in KEYS},
                       float_precision='round_trip')


def load_panel(market):
    cache = ROOT/f'.peer_pca_inputs/{market}/hidden_cache'
    original = read_csv(cache/'meta_test.csv', KEYS+['label_start_date'])
    assert original.groupby('as_of_date').label_start_date.nunique().eq(1).all()
    calendar = original.groupby('as_of_date').label_start_date.first()
    assert calendar.index[1:].tolist() == calendar.iloc[:-1].tolist()
    archive = ROOT/f'.peer_pca_inputs/{market}/predictions/Prediction_online_seed100.csv.gz'
    target_cols = ['actual_return_1', 'actual_return_10']
    original = original.merge(read_csv(archive, KEYS+target_cols), on=KEYS, validate='one_to_one')
    assert len(original) == len(np.load(cache/'current_test.npy'))
    extra_file = (OLD if market=='csi300' else RUN)/f'dasfaa_extra_features_{market}.npz'
    full = original
    if market != 'nasdaq100':
        with np.load(extra_file) as extra:
            added = pd.DataFrame({'as_of_date':extra['dates'], 'stock':extra['stocks'],
                                  target_cols[0]:extra['labels'][:,0],
                                  target_cols[1]:extra['labels'][:,-1]})
        full = pd.concat([original, added], ignore_index=True)
    assert len(full) == LIMITS[market][2] and not full.duplicated(KEYS).any()
    return original, full, calendar


def load_quotes(market, full, calendar):
    raw_dir = unique_path(DATA, '*/' + RAW[market])
    dates = pd.Index([calendar.index[0], *calendar], name='date')
    stocks = set(full.stock)
    panels = {}
    for path in sorted(raw_dir.glob('*.csv')):
        if market == 'csi300':
            stock = path.stem.replace('_', '.')
        elif market == 'sp500':
            stock = 'us.' + path.stem.removeprefix('us_').replace('_','-')
        else:
            stock = path.stem.removeprefix('us_')
        if stock not in stocks:
            continue
        frame = pd.read_csv(path, usecols=['date','open','close'], dtype={'date':str})
        assert not frame.date.duplicated().any(), path
        panels[stock] = frame.set_index('date').reindex(dates)
    assert set(panels) == stocks, sorted(stocks-set(panels))
    quotes = pd.concat(panels, names=['stock'])
    matrix = quotes.unstack('stock').reindex(dates)
    opens, closes = (matrix[col].where(np.isfinite(matrix[col]) & matrix[col].gt(0))
                     for col in ('open','close'))
    before = pd.MultiIndex.from_arrays([full.as_of_date,full.stock])
    after = pd.MultiIndex.from_arrays([full.as_of_date.map(calendar),full.stock])
    flat_closes = closes.stack()
    actual = flat_closes.reindex(after).to_numpy()/flat_closes.reindex(before).to_numpy()-1
    expected = full.actual_return_1.to_numpy()
    if market=='csi300':
        # The archived CSI replay replaces some unobserved next-close returns by zero.
        take = np.isfinite(actual)
    else:
        take = np.isfinite(actual) & np.isfinite(expected)
    print(market, 'matching price pairs', take.sum(), 'of', len(take), flush=True)
    np.testing.assert_allclose(actual[take], expected[take], atol=6e-7, rtol=0)
    return opens, closes


def evaluate(market):
    original, full, calendar = load_panel(market)
    opens, closes = load_quotes(market, full, calendar)
    top, drop, _, days = LIMITS[market]
    assert len(calendar) == days
    cfg = dict(top_k=top, drop_n=drop, min_hold=5)
    results, paths = [], []
    score_dir = OLD if market=='csi300' else RUN/market

    def record(name, seed, pool, panel, score):
        merged = panel.merge(score, on=KEYS, validate='one_to_one')
        assert len(merged) == len(panel) and np.isfinite(merged.score).all(), (name, seed)
        observed = merged.loc[np.isfinite(merged.actual_return_10)]
        ranks = pd.Series({date:spearmanr(g.score,g.actual_return_10).statistic
                           for date,g in observed.groupby('as_of_date')})
        assert ranks.index.tolist() == calendar.index.tolist() and np.isfinite(ranks).all()
        _, _, holdings = evaluate_scores(merged[KEYS], merged.score.to_numpy(),
                                         np.zeros((len(merged),10)), cfg)
        targets = holdings.pivot(index='as_of_date', columns='stock', values='weight')
        targets = targets.reindex(index=calendar.index, columns=opens.columns).fillna(0.)
        assert np.allclose(targets.sum(axis=1),1.) and targets.gt(0).sum(axis=1).eq(top).all()
        daily = run_open(targets, calendar, opens, closes)
        daily['rankic'] = daily.as_of_date.map(ranks)
        tags = dict(market=market, pool=pool, model=name, seed=seed)
        results.append(dict(**tags, stock_dates=len(panel), rankic=ranks.mean(),
                            **portfolio_statistics(daily.net_return)))
        paths.append(daily.assign(**tags))

    for seed in SEEDS:
        for name in MODELS:
            scores = read_csv(score_dir/f"endpoint_{seed}_{name.replace(' ','_')}.csv.gz")
            record(name,seed,'full',full,scores)
            if market == 'csi300' and name == 'Full Model':
                record(name,seed,'shared',original,scores)
    if market != 'sp500':
        pattern = '*cost_aligned*' if market=='csi300' else '*nasdaq100_paper_data*'
        prediction_dir, = (ARCHIVED_RESULTS / market).glob(f'*/{pattern}/predictions')
        for name, stem in EXTERNAL.items():
            zero_shot = (prediction_dir/f'{stem}.csv.gz').is_file()
            for seed in ((0,) if zero_shot else SEEDS):
                suffix = '' if zero_shot else f'_seed{seed}'
                model_scores = read_csv(prediction_dir/f'{stem}{suffix}.csv.gz', KEYS+['pred_return_10'])
                panel = original if market=='csi300' else full
                record(name, seed, 'shared' if market=='csi300' else 'full', panel,
                       model_scores.rename(columns={'pred_return_10':'score'}))
                if market == 'csi300' and stem=='MASTER_OHLCV':
                    record(name,seed,'full',full,read_csv(RUN/market/f'master_{seed}.csv.gz'))

    daily = pd.concat(paths,ignore_index=True)
    summary = pd.DataFrame(results)
    paired = []
    for metric in ('net_return','rankic'):
        wide = daily.query("pool=='full' and model in @MODELS").pivot(
            index='as_of_date', columns=['model','seed'],values=metric)
        ids = block_indices(len(wide),10)
        for ref in MODELS[:2]:
            a,b = wide['Full Model'].to_numpy(), wide[ref].to_numpy()
            if metric=='net_return':
                draws = (np.prod(1+a[ids],axis=1)-np.prod(1+b[ids],axis=1)).mean(axis=1)
                delta = (np.prod(1+a,axis=0)-np.prod(1+b,axis=0)).mean()
            else:
                draws = (a[ids]-b[ids]).mean(axis=(1,2))
                delta = (a-b).mean()
            low,high = np.percentile(draws,[2.5,97.5])
            paired.append(dict(market=market,metric=metric,reference=ref,delta=delta,low=low,high=high))
    old = RUNS / 'historical_daily'
    ranks = daily.query("pool=='full' and model in @MODELS").sort_values(['model','seed','as_of_date'])
    published = read_csv(old/f'{market}_daily.csv').sort_values(['model','seed','as_of_date'])
    np.testing.assert_array_equal(ranks.as_of_date, published.as_of_date)
    np.testing.assert_allclose(ranks.rankic, published.rankic, atol=1e-7,rtol=0)
    if market == 'csi300':
        previous = read_csv(OLD/'daily.csv').sort_values(['model','seed','as_of_date'])
        np.testing.assert_allclose(ranks.net_return, previous.net_return, atol=1e-12,rtol=0)
    daily.to_csv(RUN/market/'open_daily.csv',index=False)
    summary.to_csv(RUN/market/'open_by_seed.csv',index=False)
    grouped = summary.groupby(['market','pool','model'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    grouped.to_csv(RUN/market/'open_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(RUN/market/'open_intervals.csv',index=False)
    print(grouped.to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('market', choices=LIMITS)
    evaluate(parser.parse_args().market)
