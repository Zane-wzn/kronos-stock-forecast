"""Fit a frozen-feature linear head and re-evaluate archived scores."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd

from scripts.evaluation.paired_statistics import block_indices
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch
from torch.nn.functional import smooth_l1_loss

from ..paths import ARCHIVED_RESULTS, DATA, unique_path
from .replay import HERE, CSI, US, ROOT, KEYS, LIMITS, MARKETS, rp, csi
from scripts.finca.next_open_main_20261123.replay_markets import EXTERNAL, load_panel, load_quotes
from scripts.finca.csi300_adjusted_20261124.accounting import run_open
from scripts.evaluation.evaluate_rank_scores import evaluate_scores, portfolio_statistics


def fit_heads(market, source):
    if all((HERE / market / f'linear_seed{seed}.pt').is_file() for seed in rp.SEEDS):
        return
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    if market == 'csi300':
        validation = unique_path(DATA, '*/csi300/*/reference/hidden_cache')
    else:
        validation = unique_path(DATA, '*/sp500/共享资源/aligned_235d/reference/hidden_cache')
    x, y, vx, vy = [torch.from_numpy(np.load(p)).cuda() for p in
                    (cache/'hidden_train.npy', source/'labels_train.npy',
                     validation/'hidden_val.npy', source/'labels_val.npy')]
    dates = pd.read_csv(cache/'meta_train.csv', usecols=['as_of_date']).as_of_date
    groups = list(dates.groupby(dates, sort=True).indices.values())
    size = 32 if market == 'csi300' else 16
    def batches(seed):
        rng = np.random.default_rng(seed)
        rows = [chunk for group in groups for chunk in np.array_split(rng.permutation(group),
                np.arange(size, len(group), size))]
        return [rows[i] for i in rng.permutation(len(rows))]
    for seed in rp.SEEDS:
        dest = HERE / market / f'linear_seed{seed}.pt'
        if dest.exists():
            continue
        rp.seed_all(seed)
        torch.set_num_threads(1)
        model = rp.new_model().head.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4,
                                      fused=True, capturable=True)
        indices = torch.zeros(size, device='cuda', dtype=torch.long)
        mask = torch.ones(size, device='cuda')
        def step():
            optimizer.zero_grad(set_to_none=True)
            errors = smooth_l1_loss(model(x[indices]), y[indices], reduction='none')
            loss = (errors.mean(1) * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        initial = [p.detach().clone() for p in model.parameters()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        with torch.no_grad():
            for p, value in zip(model.parameters(), initial):
                p.copy_(value)
            for state in optimizer.state.values():
                for field in state.values():
                    field.zero_()
        if market == 'csi300':
            val_dates = pd.read_csv(cache/'meta_val.csv', usecols=['as_of_date']).as_of_date
            val_groups = list(val_dates.groupby(val_dates, sort=False).indices.values())
            val_batches = [part for group in val_groups for part in np.array_split(
                group, np.arange(size,len(group),size))]
        best, stale = np.inf, 0
        for epoch in range(1, 6):
            rows = batches(seed+epoch)
            keys = np.zeros((len(rows),size), dtype=np.int64)
            weights = np.zeros_like(keys,dtype=np.float32)
            for i, row in enumerate(rows):
                keys[i,:len(row)] = row
                weights[i,:len(row)] = 1.
            keys, weights = torch.from_numpy(keys).cuda(), torch.from_numpy(weights).cuda()
            for ix, weight in zip(keys,weights):
                indices.copy_(ix)
                mask.copy_(weight)
                graph.replay()
            with torch.inference_mode():
                predictions = model(vx)
                loss = (torch.stack([smooth_l1_loss(predictions[ix], vy[ix]) for ix in val_batches]).mean()
                        if market == 'csi300' else smooth_l1_loss(predictions, vy)).item()
            assert np.isfinite(loss)
            if loss < best:
                state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
                best, stale = loss, 0
            else:
                stale += 1
            print(market, seed, epoch, loss, flush=True)
            if stale == 2:
                break
        torch.save(state, dest)


def shared_intervals(daily):
    wide = daily.pivot(index='as_of_date', columns=['model', 'seed'], values='net_return').sort_index()
    assert len(wide) == LIMITS['csi300'][-1] and wide.notna().all().all()
    returns = {
        model: wide[model].sort_index(axis=1).to_numpy()
        for model in ('Full Model', 'Static Adapter', 'Kronos (zero-shot)')
    }
    results = []
    for width in (5, 10, 20):
        positions = block_indices(len(wide), width)
        for treatment, control in (
            ('Full Model', 'Static Adapter'),
            ('Full Model', 'Kronos (zero-shot)'),
            ('Static Adapter', 'Kronos (zero-shot)'),
        ):
            first, second = (returns[model] for model in (treatment, control))
            exponent = 252 / len(wide)
            delta = (np.prod(1 + first, axis=0) ** exponent - np.prod(1 + second, axis=0) ** exponent).mean()
            draws = (np.prod(1 + first[positions], axis=1) ** exponent
                     - np.prod(1 + second[positions], axis=1) ** exponent).mean(axis=1)
            low, high = np.percentile(100 * draws, [2.5, 97.5])
            results.append(dict(treatment=treatment, control=control, block_days=width,
                                delta=100 * delta, low=low, high=high))
    return pd.DataFrame(results)


def evaluate(market):
    source = CSI if market == 'csi300' else US / market
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    original, full, calendar = load_panel(market)
    if market == 'csi300':
        opens, closes = load_quotes(market, full, calendar)
        quotes = (opens, closes, *csi.action_matrices(opens))
    else:
        quotes = tuple(pd.read_csv(source / f'{name}.csv', index_col=0)
                       for name in ('opens', 'closes', 'dividends', 'splits'))
    top,drop,_,days = LIMITS[market]
    actual = np.load(source/'labels_test.npy')[:,-1]
    tests = original[KEYS].assign(actual=actual)
    pattern = {'csi300': '*cost_aligned*', 'sp500': '*sp500_paper_data*',
               'nasdaq100': '*nasdaq100_paper_data*'}[market]
    folder = unique_path(ARCHIVED_RESULTS / market, f'*/{pattern}/predictions')
    rows = []
    shared_paths = []

    def record(task, name, seed, scores, panel=tests):
        merged = panel.merge(scores, on=KEYS, how='left', validate='one_to_one')
        assert np.isfinite(merged.score).all()
        valid = merged.dropna(subset='actual')
        rank = valid.groupby('as_of_date')[['score','actual']].apply(
            lambda g: spearmanr(g.score,g.actual).statistic)
        assert np.isfinite(rank).all() and len(rank)==days
        _,_,hold = evaluate_scores(merged[KEYS],merged.score.to_numpy(),np.zeros((len(merged),10)),
                                   dict(top_k=top,drop_n=drop,min_hold=5))
        weights = hold.pivot(index='as_of_date', columns='stock', values='weight')
        weights = weights.reindex(index=calendar.index,columns=quotes[0].columns).fillna(0.)
        assert np.allclose(weights.sum(axis=1),1.)
        path = run_open(weights,calendar,*quotes)
        if market == 'csi300' and (
            (task == 'external' and name in ('Full Model', 'Kronos (zero-shot)'))
            or (task == 'sensitivity' and name == 'Static Adapter')
        ):
            shared_paths.append(path[['as_of_date', 'net_return']].assign(model=name, seed=seed))
        if task == 'sensitivity':
            return
        stats = portfolio_statistics(path.net_return)
        rows.append(dict(market=market,task=task,model=name,seed=seed,n=len(merged),rankic=rank.mean(),
                         annualized_return=(1+stats['net_cumulative_return'])**(252/days)-1, **stats))

    def external_scores(file):
        return pd.read_csv(file,usecols=KEYS+['pred_return_10']).rename(columns={'pred_return_10':'score'})

    offline = market in ('csi300','sp500')
    if offline:
        fit_heads(market,source)
        frozen = torch.from_numpy(np.load(cache/'hidden_test.npy'))
        record('offline','Kronos (zero-shot)',0,external_scores(folder/'Kronos_zero_shot.csv.gz'))
        for seed in rp.SEEDS:
            head = torch.nn.Linear(512,10)
            head.load_state_dict(torch.load(HERE/market/f'linear_seed{seed}.pt',weights_only=True))
            with torch.inference_mode():
                direct = head(frozen).numpy()[:,-1]
                adapter = rp.make_model(source/f'seed{seed}/model.pt',False)
                adapted = rp.predict_tensor(adapter,frozen,4096)[:,-1]
            record('offline','Linear head',seed,tests[KEYS].assign(score=direct))
            record('offline','Adapter + head',seed,tests[KEYS].assign(score=adapted))
    if market != 'sp500':
        for name,stem in EXTERNAL.items():
            plain=folder/f'{stem}.csv.gz'
            for seed in ((0,) if plain.exists() else rp.SEEDS):
                path=plain if seed==0 else folder/f'{stem}_seed{seed}.csv.gz'
                record('external',name,seed,external_scores(path))
        for seed in rp.SEEDS:
            columns = ['Full Model', 'Static Adapter'] if market == 'csi300' else ['Full Model']
            scores = pd.read_csv(HERE / market / f'endpoint_seed{seed}.csv.gz', usecols=KEYS + columns)
            record('external', 'Full Model', seed, scores[KEYS + ['Full Model']].rename(columns={'Full Model': 'score'}))
            if market == 'csi300':
                static = scores[KEYS + ['Static Adapter']].rename(columns={'Static Adapter': 'score'})
                record('sensitivity', 'Static Adapter', seed, static)
    if shared_paths:
        daily = pd.concat(shared_paths, ignore_index=True)
        daily.to_csv(HERE / market / 'shared_candidate_daily.csv', index=False)
        shared_intervals(daily).to_csv(HERE / market / 'shared_candidate_intervals.csv', index=False)
    pd.DataFrame(rows).to_csv(HERE/market/'adjusted_comparisons.csv',index=False)
    print(pd.DataFrame(rows).groupby(['task','model'])[['rankic','annualized_return','sharpe','max_drawdown']].mean())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('market',choices=MARKETS)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        evaluate(parser.parse_args().market)
