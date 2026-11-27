"""Rebuild the paper ablation tables and curves under adjusted accounting."""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from ..paths import OUTPUT as HERE, ROOT, RUNS
from scripts.finca.csi300_adjusted_20261124 import evaluate as csi

CSI = RUNS / 'csi300_adjusted_20261124'
US = RUNS / 'us_adjusted_20261125'
ARCHIVE = RUNS / 'next_open_main_20261123'
from scripts.finca.csi300_adjusted_20261124.accounting import run_open
from scripts.finca.next_open_main_20261123.replay_markets import KEYS, LIMITS, load_panel, load_quotes
from scripts.evaluation.evaluate_rank_scores import evaluate_scores, portfolio_statistics
from scripts.evaluation.paired_statistics import block_indices
from scripts.experiments import peer_pca_correction as pc, peer_pca_replay as rp
from src.kronos_continual.online_peer_pca_head import OnlinePCA, weighted_feature_mean

MODELS = ('Static Adapter', 'Online Head Only', 'Full Model')
VARIANTS = ('Target only', 'Unprojected peers', 'Fixed PCA basis', 'Online PCA')
MARKETS = ('csi300', 'nasdaq100', 'sp500')


def input_data(market):
    source = CSI if market == 'csi300' else US / market
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    _, full, calendar = load_panel(market)
    panels = [pd.read_csv(cache / f'meta_{split}.csv',
                          usecols=lambda c: c in KEYS + ['label_dates', 'label_end_date'])
              for split in ('train', 'val', 'test')]
    train, _, test = panels
    meta = full[KEYS].copy()
    np.testing.assert_array_equal(test[KEYS].to_numpy(), meta.iloc[:len(test)].to_numpy())
    features = np.load(cache / 'hidden_test.npy')
    labels = np.load(source / 'labels_test.npy')
    if market != 'nasdaq100':
        folder = RUNS / 'csi300_next_open_20261123' if market == 'csi300' else ARCHIVE
        with np.load(folder / f'dasfaa_extra_features_{market}.npz') as extra:
            np.testing.assert_array_equal(meta.iloc[len(test):].to_numpy(),
                                          np.column_stack([extra['dates'], extra['stocks']]))
            features = np.concatenate([features, extra['hidden']])
        filename = 'labels_test_extra.npy' if market == 'csi300' else 'labels_extra.npy'
        labels = np.concatenate([labels, np.load(source / filename)])
    groups = meta.groupby('as_of_date', sort=True).indices
    if market == 'csi300':
        mapping = test.drop_duplicates('as_of_date').set_index('as_of_date').label_dates
        meta['label_dates'] = meta.as_of_date.map(mapping)
        schedule = csi.build_schedule(meta, labels, train, groups)
        opens, closes = load_quotes(market, full, calendar)
        quotes = (opens, closes, *csi.action_matrices(opens))
    else:
        schedule = {date: ids for date, ids in rp.make_schedule(test).items() if len(ids)}
        assert len(schedule) == 225
        assert sum(map(len, schedule.values())) == {'nasdaq100': 22050, 'sp500': 112454}[market]
        assert all(np.isfinite(labels[ids]).all() for ids in schedule.values())
        quotes = tuple(pd.read_csv(source / f'{name}.csv', index_col=0)
                       for name in ('opens', 'closes', 'dividends', 'splits'))
    panels[-1] = meta
    adjusted = pd.read_csv(source / 'adjusted_close.csv.gz', index_col=0)
    stocks, weights = csi.peer_weights(adjusted, panels)
    training_hidden = np.load(cache / 'hidden_train.npy', mmap_mode='r')
    assert len(features) == len(labels) == len(meta)
    return source, meta, calendar, quotes, groups, schedule, stocks, weights, train, training_hidden, torch.from_numpy(features), torch.from_numpy(labels)


def update(market, model, optimizer, features, labels, batch):
    if market != 'csi300':
        rp.daily_update(model, optimizer, features[batch], labels[batch], 256)
        return
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack([F.smooth_l1_loss(model(features[rows])[:, k], labels[rows, k])
                        for k, rows in enumerate(batch) if len(rows)]).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad),
                                   1, error_if_nonfinite=True)
    optimizer.step()


def score(market, checkpoint, context, name, features, labels, groups, stock_ids, weights, schedule):
    model = rp.make_model(checkpoint, name != MODELS[0])
    params = [dict(params=model.head.parameters(), lr=1e-4)]
    joint = name == MODELS[-1]
    if joint:
        model.adapter.requires_grad_(True)
        params.append(dict(params=model.adapter.parameters(), lr=1e-5))
    optimizer = torch.optim.AdamW(params, weight_decay=0) if name != MODELS[0] else None
    pca = OnlinePCA(context.mean, context.basis, context.eigen, decay=2 ** (-1 / pc.HALF_LIFE))
    scores = np.empty((len(features), len(VARIANTS) if joint else 1))
    for day, ids in groups.items():
        with torch.no_grad():
            encoded = model.adapter(features[ids])
            base = model.head(encoded).numpy()[:, -1]
            head = model.head.weight.detach().numpy().astype(float)
            ids_stock = stock_ids[ids]
            points, counts = weighted_feature_mean(
                encoded.numpy().astype(float), ~np.eye(len(ids), dtype=bool),
                weights[day][np.ix_(ids_stock, ids_stock)])
            available = counts > 0
            online = base + .025 * (pca.reconstruct(points, available) @ head.T)[:, -1]
            scores[ids, -1] = online
            if joint:
                centered = np.where(available[:, None], points - pca.mean, 0.)
                fixed = (centered @ context.basis) @ context.basis.T
                scores[ids, :3] = np.column_stack(
                    [base, base + .025 * (centered @ head.T)[:, -1],
                     base + .025 * (fixed @ head.T)[:, -1]])
            pca.update(points, available)
        if optimizer is not None and day in schedule:
            update(market, model, optimizer, features, labels, schedule[day])
    assert np.isfinite(scores).all()
    return scores


def replay(market):
    source, meta, calendar, quotes, groups, schedule, stocks, weights, train, train_hidden, features, labels = input_data(market)
    original_scenario = 'adjusted_labels_and_account' if market == 'csi300' else 'adjusted'
    original = pd.read_csv(source / 'daily.csv').query('scenario == @original_scenario')
    top, drop, count, days = LIMITS[market]
    assert len(meta) == count and len(calendar) == days
    config = dict(top_k=top, drop_n=drop, min_hold=5)
    stock_ids = meta.stock.map(stocks).to_numpy(dtype=int)
    actual = labels.numpy()[:, -1]
    out = HERE / market
    out.mkdir(parents=True, exist_ok=True)
    metrics, paths = [], []

    def record(experiment, name, seed, values, reference=None):
        _, _, holdings = evaluate_scores(meta[KEYS], values, np.zeros(labels.shape), config)
        targets = (holdings.pivot(index='as_of_date', columns='stock', values='weight')
                   .reindex(index=calendar.index, columns=quotes[0].columns).fillna(0.))
        assert targets.gt(0).sum(axis=1).eq(top).all()
        daily = run_open(targets, calendar, *quotes)
        ranks = {}
        for day, ids in groups.items():
            use = ids[np.isfinite(actual[ids])]
            ranks[day] = spearmanr(values[use], actual[use]).statistic
        daily['rankic'] = daily.as_of_date.map(ranks)
        if reference:
            old = original.query('model == @reference and seed == @seed').set_index('as_of_date')
            np.testing.assert_allclose(daily[['net_return', 'rankic']],
                                       old.loc[daily.as_of_date, ['net_return', 'rankic']], atol=1e-12, rtol=0)
            if market != 'csi300':
                file = source / f'endpoint_{seed}_{reference.replace(" ", "_")}.csv.gz'
                archived = meta[KEYS].merge(pd.read_csv(file), on=KEYS, how='left', validate='one_to_one')
                np.testing.assert_allclose(values, archived.score, atol=1e-12, rtol=0)
        valid = np.isfinite(actual)
        residual = values[valid] - actual[valid]
        stats = portfolio_statistics(daily.net_return)
        fields = dict(market=market, experiment=experiment, model=name, seed=seed)
        metrics.append(dict(**fields, **stats, annualized_return=(1 + stats['net_cumulative_return']) ** (252 / days) - 1,
                            rankic=daily.rankic.mean(), rmse=np.sqrt(np.mean(residual ** 2)),
                            mae=np.mean(np.abs(residual)), da=np.mean((values[valid] > 0) == (actual[valid] > 0)),
                            observed_endpoints=valid.sum(), stock_dates=count))
        paths.append(daily.assign(**fields))

    for seed in rp.SEEDS:
        print(market, seed, flush=True)
        checkpoint = source / f'seed{seed}/model.pt'
        initial = rp.make_model(checkpoint, False)
        context = pc.fit_training_context(train, train_hidden, initial.adapter)
        endpoint = meta[KEYS].assign(actual_return_10=actual)
        for name in MODELS:
            values = score(market, checkpoint, context, name, features, labels, groups, stock_ids, weights, schedule)
            record('online_updates', name, seed, values[:, -1], reference=name)
            endpoint[name] = values[:, -1]
            if name == MODELS[-1]:
                for variant, column in zip(VARIANTS, values.T):
                    record('peer_correction', variant, seed, column, reference=name if variant == VARIANTS[-1] else None)
                    endpoint[variant] = column
        endpoint.to_csv(out / f'endpoint_seed{seed}.csv.gz', index=False)
    pd.DataFrame(metrics).to_csv(out / 'by_seed.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / 'daily.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('market', choices=MARKETS)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        replay(parser.parse_args().market)
