"""Compare archived and adjusted CSI300 policies using matched test dates."""
import argparse
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from .accounting import run_open
from .prepare import ARCHIVE, CACHE, HERE, KEYS, values_at
from ..paths import DATA, RUNS, unique_path

from scripts.evaluation.evaluate_rank_scores import evaluate_scores, portfolio_statistics
from scripts.evaluation.paired_statistics import block_indices
from scripts.experiments import peer_pca_correction as pca, peer_pca_replay as replay
from src.kronos_continual.online_peer_pca_head import cross_section_weights
from scripts.finca.next_open_main_20261123.open_execution import run_open as run_unadjusted
from scripts.finca.next_open_main_20261123.replay_markets import load_panel, load_quotes

MODELS = ('Static Adapter', 'Online Head Only', 'Full Model')
BASELINE = RUNS / 'next_open_main_20261123/csi300/open_daily.csv'


def build_schedule(meta, labels, train, groups):
    dates = list(groups)
    key = lambda stock: hashlib.sha256(('prefix-pilot-v1|' + stock).encode()).hexdigest()
    pool = set(sorted(train.stock.unique(), key=key)[:60])
    names = meta.stock.to_numpy()
    dated_path = meta.drop_duplicates('as_of_date').set_index('as_of_date').label_dates.str.split('|')
    lookup = [{path[k]: origin for origin, path in dated_path.items()} for k in range(10)]
    complete = np.isfinite(labels).all(axis=1)
    full = {day: {names[i]: i for i in ids if complete[i] and names[i] in pool}
            for day, ids in groups.items()}
    schedule = {}
    for today in dates[10:]:
        origin = lookup[-1][today]
        rows_by_horizon = []
        for k in range(10):
            prefix = lookup[k][today]
            eligible = {names[i] for i in groups[prefix] if names[i] in pool and np.isfinite(labels[i, k])}
            picked = sorted(set(full[origin]) & eligible, key=key)[:32]
            rows_by_horizon.append([full[origin][stock] for stock in picked])
        schedule[today] = rows_by_horizon
    assert len(schedule) == 216 and sum(len(rows) for day in schedule.values() for rows in day) == 69120
    return schedule


def peer_weights(adjusted, splits):
    parts = [frame[KEYS].assign(close=values_at(adjusted, frame.as_of_date, frame.stock))
             for frame in splits]
    price = pd.concat(parts).pivot(index='as_of_date', columns='stock', values='close').sort_index()
    returns = price.pct_change(fill_method=None)
    train, val, test = splits
    returns.loc[[val.as_of_date.min(), test.as_of_date.min()]] = np.nan
    bounds = returns.loc[train.as_of_date.min():train.as_of_date.max()].quantile([.01, .99])
    returns = returns.clip(lower=bounds.loc[.01], upper=bounds.loc[.99], axis=1)
    values = returns.to_numpy()
    weights = {}
    for day in sorted(test.as_of_date.unique()):
        index = returns.index.get_loc(day)
        weights[day] = cross_section_weights(values[max(0, index - 60):index], 40)
    return {s: i for i, s in enumerate(returns.columns)}, weights


def predictions(seed, meta, features, labels, train, groups, weights, schedule, raw_refit=False):
    checkpoint = HERE / (f'raw_seed{seed}' if raw_refit else f'seed{seed}') / 'model.pt'
    if raw_refit and not checkpoint.exists():
        checkpoint = HERE / f'raw_parallel_seed{seed}' / 'model.pt'
    training_features = np.load(CACHE / 'hidden_train.npy', mmap_mode='r')
    context = pca.fit_training_context(train, training_features, replay.make_model(checkpoint, False).adapter)
    for name in MODELS:
        model = replay.make_model(checkpoint, name != MODELS[0])
        if name == MODELS[2]:
            model.adapter.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        learning_rates = [dict(params=model.head.parameters(), lr=1e-4)]
        if name == MODELS[2]:
            learning_rates.append(dict(params=model.adapter.parameters(), lr=1e-5))
        optimizer = torch.optim.AdamW(learning_rates, weight_decay=0) if trainable else None
        observe, correction, counts, updates = pca.correction_observer(meta, *weights, context)
        scores = np.empty(len(meta))
        for date, ids in groups.items():
            with torch.inference_mode():
                encoded = model.adapter(features[ids])
                scores[ids] = model.head(encoded).numpy()[:, -1]
                observe(date, ids, encoded, model.head)
            if date not in schedule or optimizer is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([F.smooth_l1_loss(model(features[rows])[:, k], labels[rows, k])
                                for k, rows in enumerate(schedule[date]) if len(rows)]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
            optimizer.step()
        pca.verify_corrections(correction, counts)
        assert len(updates) == len(groups)
        yield name, scores + .025 * correction[:, -1]


def action_matrices(opens):
    rows = pd.read_csv(HERE / 'corporate_actions.csv', dtype={'stock': str, 'date': str})
    matrices = {}
    for field in ('dividend', 'share_bonus'):
        matrices[field] = (rows.pivot(index='date', columns='stock', values=field)
                            .reindex(index=opens.index, columns=opens.columns).fillna(0.))
    return matrices['dividend'], 1 + matrices['share_bonus']


def paired(daily):
    rows = []
    for scenario, frame in daily.groupby('scenario'):
        series = frame.pivot(index='as_of_date', columns=['model', 'seed'], values='net_return').sort_index()
        days = len(series)
        assert days == 226 and np.isfinite(series.to_numpy()).all()
        annualize = lambda x, axis: np.prod(1 + x, axis=axis) ** (252 / days) - 1
        full = series[MODELS[2]].to_numpy()
        for name in MODELS[:2]:
            reference = series[name].to_numpy()
            point = (annualize(full, 0) - annualize(reference, 0)).mean()
            for block in (10, 20, 40):
                indices = block_indices(days, block)
                differences = (annualize(full[indices], 1) - annualize(reference[indices], 1)).mean(axis=1)
                low, high = np.percentile(differences, [2.5, 97.5])
                rows.append(dict(scenario=scenario, reference=name, block_days=block, delta=point, low=low, high=high))
    return pd.DataFrame(rows)


def main(archived_only, raw_refit=False):
    _, full, calendar = load_panel('csi300')
    opens, closes = load_quotes('csi300', full, calendar)
    dividends, multipliers = action_matrices(opens)
    panels = [pd.read_csv(CACHE / f'meta_{s}.csv', usecols=KEYS + ['label_dates']) for s in ('train', 'val', 'test')]
    train, _, test = panels
    with np.load(ARCHIVE / 'dasfaa_extra_features_csi300.npz') as extra:
        added = pd.DataFrame(dict(as_of_date=extra['dates'], stock=extra['stocks']))
        added['label_dates'] = added.as_of_date.map(test.drop_duplicates('as_of_date').set_index('as_of_date').label_dates)
        meta = pd.concat([test, added], ignore_index=True)
        features = torch.from_numpy(np.concatenate([np.load(CACHE / 'hidden_test.npy'), extra['hidden']]))
    panels[-1] = meta
    labels = np.concatenate([np.load(HERE / 'labels_test.npy'), np.load(HERE / 'labels_test_extra.npy')])
    np.testing.assert_array_equal(meta[KEYS].to_numpy(), full[KEYS].to_numpy())
    groups = meta.groupby('as_of_date', sort=True).indices
    baseline = pd.read_csv(BASELINE).query("pool == 'full' and model in @MODELS").assign(scenario='paper_raw')
    daily, metrics = [baseline], []
    for key, group in baseline.groupby(['scenario', 'model', 'seed']):
        metrics.append(dict(zip(['scenario', 'model', 'seed'], key), rankic=group.rankic.mean(),
                            **portfolio_statistics(group.net_return)))
    weights = schedule = None
    if not archived_only and not raw_refit:
        weights = peer_weights(pd.read_csv(HERE / 'adjusted_close.csv.gz', index_col=0, dtype={'date': str}), panels)
        schedule = build_schedule(meta, labels, train, groups)

    def record(scenario, seed, name, score, target_labels=labels):
        _, _, held = evaluate_scores(meta[KEYS], score, np.zeros_like(labels),
                                     dict(top_k=30, drop_n=3, min_hold=5))
        targets = (held.pivot(index='as_of_date', columns='stock', values='weight')
                   .reindex(index=calendar.index, columns=opens.columns).fillna(0.))
        if scenario == 'adjusted_account_only':
            old = baseline[(baseline.seed == seed) & (baseline.model == name)].set_index('as_of_date')
            raw_replayed = run_unadjusted(targets, calendar, opens, closes)
            expected = old.loc[raw_replayed.as_of_date, 'net_return']
            np.testing.assert_allclose(raw_replayed.net_return, expected, atol=1e-12, rtol=0)
        if scenario == 'raw_refit':
            result = run_unadjusted(targets, calendar, opens, closes).assign(cash_dividend=0.)
        else:
            result = run_open(targets, calendar, opens, closes, dividends, multipliers)
        ranks = {}
        for date, ids in groups.items():
            selected = ids[np.isfinite(target_labels[ids, -1])]
            ranks[date] = spearmanr(score[selected], target_labels[selected, -1]).statistic
        result['rankic'] = result.as_of_date.map(ranks)
        info = dict(scenario=scenario, seed=seed, model=name)
        summary = dict(**info, rankic=result.rankic.mean(), dividends=result.cash_dividend.sum(),
                       **portfolio_statistics(result.net_return))
        metrics.append(summary)
        daily.append(result.assign(**info))
        print(summary, flush=True)

    for seed in replay.SEEDS if not raw_refit else ():
        for name in MODELS:
            archive = pd.read_csv(ARCHIVE / f'endpoint_{seed}_{name.replace(" ", "_")}.csv.gz')
            score = meta[KEYS].merge(archive, on=KEYS, how='left', validate='one_to_one').score.to_numpy()
            assert np.isfinite(score).all()
            record('adjusted_account_only', seed, name, score)
        if not archived_only:
            for name, score in predictions(seed, meta, features, torch.from_numpy(labels), train, groups, weights, schedule):
                record('adjusted_labels_and_account', seed, name, score)
    if raw_refit:
        raw = unique_path(DATA, '*/csi300_daily/csi300_daily')
        closes_by_stock = {
            path.stem.replace('_', '.'): pd.read_csv(path, usecols=['date', 'close']).set_index('date').close
            for path in sorted(raw.glob('*.csv'))
        }
        weights = peer_weights(pd.DataFrame(closes_by_stock).sort_index(), panels)
        with np.load(ARCHIVE / 'dasfaa_extra_features_csi300.npz') as extra:
            raw_labels = np.concatenate([np.load(CACHE / 'labels_test.npy'), extra['labels']])
        schedule = build_schedule(meta, raw_labels, train, groups)
        for seed in replay.SEEDS:
            for name, score in predictions(seed, meta, features, torch.from_numpy(raw_labels), train, groups,
                                           weights, schedule, raw_refit=True):
                record('raw_refit', seed, name, score, target_labels=raw_labels)
                record('raw_refit_adjusted_account', seed, name, score)
        daily = [pd.read_csv(HERE / 'daily.csv'), *daily[1:]]
        raw_metrics = [row for row in metrics if row['scenario'] != 'paper_raw']
        metrics = [*pd.read_csv(HERE / 'by_seed.csv').to_dict('records'), *raw_metrics]
    daily = pd.concat(daily, ignore_index=True)
    suffix = '_pilot' if archived_only else '_with_raw_refit' if raw_refit else ''
    daily.to_csv(HERE / f'daily{suffix}.csv', index=False)
    metrics = pd.DataFrame(metrics)
    metrics['annualized_return'] = [(1 + value) ** (252 / len(calendar)) - 1
                                    for value in metrics.net_cumulative_return]
    metrics.to_csv(HERE / f'by_seed{suffix}.csv', index=False)
    fields = ['annualized_return', 'rankic', 'sharpe', 'max_drawdown']
    summary = metrics.groupby(['scenario', 'model'])[fields].mean().reset_index()
    summary.to_csv(HERE / f'summary{suffix}.csv', index=False)
    checks = paired(daily)
    checks.to_csv(HERE / f'paired_intervals{suffix}.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print(checks.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--archived-only', action='store_true')
    group.add_argument('--raw-refit', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main(args.archived_only, args.raw_refit)
