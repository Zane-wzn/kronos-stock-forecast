"""Verify the adjusted replays and format the data for the paper."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .replay import HERE, LIMITS, MARKETS, MODELS, VARIANTS, block_indices

TITLES = dict(csi300='CSI300', nasdaq100='Nasdaq-100', sp500='S&P 500')
MODEL_ZH = dict(zip(MODELS, ('固定预测器', '仅更新预测头', '联合更新')))
PEER_ZH = dict(zip(VARIANTS, ('仅目标股', '同伴特征不投影', '固定 PCA 基', '在线 PCA')))


def intervals(daily, pairs):
    results = []
    for (market, experiment), rows in daily.groupby(['market', 'experiment']):
        for metric in ('net_return', 'rankic'):
            panel = rows.pivot(index='as_of_date', columns=['model', 'seed'], values=metric).sort_index()
            days = len(panel)
            assert days == LIMITS[market][-1] and not panel.isna().any().any()
            for treatment, control in pairs[experiment]:
                first, second = panel[treatment].to_numpy(), panel[control].to_numpy()
                for size in (10, 20, 40) if experiment == 'online_updates' else (10,):
                    positions = block_indices(days, size)
                    scale = 100 if metric == 'net_return' else 1

                    def aggregate(values, axis):
                        if metric == 'net_return':
                            return np.prod(1 + values, axis=axis) ** (252 / days) - 1
                        return values.mean(axis=axis)

                    point = (aggregate(first, 0) - aggregate(second, 0)).mean()
                    draws = (aggregate(first[positions], 1) - aggregate(second[positions], 1)).mean(axis=1)
                    low, high = np.percentile(draws, [2.5, 97.5])
                    results.append(dict(market=market, experiment=experiment,
                                        metric='annualized_return_pp' if scale == 100 else metric,
                                        treatment=treatment, control=control, block_days=size,
                                        delta=scale * point, low=scale * low, high=scale * high))
    return pd.DataFrame(results)


def verify_archive(directory):
    """Recompute metrics and paired intervals from the released daily series."""
    daily = pd.read_csv(directory / 'all_daily.csv.gz')
    seeds = pd.read_csv(directory / 'all_by_seed.csv')
    keys = ['market', 'experiment', 'model', 'seed']
    assert not daily.duplicated(keys + ['as_of_date']).any()
    assert not seeds.duplicated(keys).any() and len(seeds) == 63
    daily = daily.sort_values(keys + ['as_of_date'])
    totals = daily.groupby(keys).agg(
        sessions=('net_return', 'size'),
        wealth=('net_return', lambda daily_return: np.prod(1 + daily_return)),
        rankic=('rankic', 'mean'),
    )
    expected = seeds.set_index(keys).sort_index()
    pd.testing.assert_index_equal(totals.index, expected.index)
    np.testing.assert_allclose(totals.wealth - 1, expected.net_cumulative_return, atol=1e-12, rtol=0)
    annualized = totals.wealth ** (252 / totals.sessions) - 1
    np.testing.assert_allclose(annualized, expected.annualized_return, atol=1e-12, rtol=0)
    np.testing.assert_allclose(totals.rankic, expected.rankic, atol=1e-12, rtol=0)
    policies = [(MODELS[-1], MODELS[0]), (MODELS[-1], MODELS[1]), (MODELS[1], MODELS[0])]
    peers = [(VARIANTS[-1], name) for name in VARIANTS[:3]]
    recalculated = intervals(daily, {'online_updates': policies, 'peer_correction': peers})
    archived = pd.concat([
        pd.read_csv(directory / 'tables7_8_update_intervals.csv'),
        pd.read_csv(directory / 'table5_paired_intervals.csv'),
    ], ignore_index=True)
    group = ['market', 'experiment', 'metric', 'treatment', 'control', 'block_days']
    pd.testing.assert_frame_equal(
        recalculated.set_index(group).sort_index(),
        archived.set_index(group).sort_index(),
        atol=1e-10,
        rtol=0,
    )
    from .offline_comparisons import shared_intervals

    shared_daily = pd.read_csv(directory / 'shared_candidate_daily.csv')
    shared_expected = pd.read_csv(directory / 'shared_candidate_intervals.csv')
    shared = shared_intervals(shared_daily)
    pd.testing.assert_frame_equal(shared, shared_expected, rtol=0, atol=1e-10)
    print(f'Verified {len(seeds)} seed rows, {len(daily)} daily rows, '
          f'and {len(recalculated) + len(shared)} paired intervals.')


def markdown_table(columns, rows):
    return ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |',
            *['| ' + ' | '.join(row) + ' |' for row in rows]]


def main():
    seeds = pd.concat([pd.read_csv(HERE / m / 'by_seed.csv') for m in MARKETS], ignore_index=True)
    daily = pd.concat([pd.read_csv(HERE / m / 'daily.csv') for m in MARKETS], ignore_index=True)
    group_keys = ['market', 'experiment', 'model']
    count_keys = group_keys + ['seed']
    assert len(seeds) == 63 and not seeds.duplicated(count_keys).any()
    assert not daily.duplicated(count_keys + ['as_of_date']).any()
    for market in MARKETS:
        market_rows = daily.query('market == @market')
        counts = market_rows.groupby(count_keys).size()
        assert len(counts) == 21 and counts.eq(LIMITS[market][-1]).all()
        by_seed = seeds.query('market == @market')
        assert by_seed.observed_endpoints.eq({'csi300': 66127, 'nasdaq100': 23030, 'sp500': 117473}[market]).all()
        assert by_seed.stock_dates.eq(LIMITS[market][-2]).all()
        for seed in (100, 101, 102):
            predictions = pd.read_csv(HERE / market / f'endpoint_seed{seed}.csv.gz')
            assert len(predictions) == LIMITS[market][-2]
            assert not predictions.duplicated(['stock','as_of_date']).any()
            np.testing.assert_array_equal(predictions[MODELS[-1]], predictions[VARIANTS[-1]])
        a = by_seed.query('experiment == "online_updates" and model == @MODELS[-1]').sort_values('seed')
        b = by_seed.query('experiment == "peer_correction" and model == @VARIANTS[-1]').sort_values('seed')
        np.testing.assert_array_equal(a[['rankic','annualized_return','rmse']], b[['rankic','annualized_return','rmse']])

    metrics = ['annualized_return','net_cumulative_return','rankic','sharpe','max_drawdown','rmse','mae','da']
    summary = seeds.groupby(group_keys)[metrics].agg(['mean','std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()
    online = summary.query('experiment == "online_updates"').copy()
    peer = summary.query('experiment == "peer_correction"').copy()
    online_pairs = [(MODELS[-1], MODELS[0]), (MODELS[-1], MODELS[1]), (MODELS[1], MODELS[0])]
    peer_pairs = [(VARIANTS[-1], name) for name in VARIANTS[:3]]
    paired = intervals(daily, {'online_updates': online_pairs, 'peer_correction': peer_pairs})
    assert len(paired) == 72
    online.to_csv(HERE / 'table4_online_updates.csv', index=False)
    peer.to_csv(HERE / 'table5_peer_correction.csv', index=False)
    online[['market','model','rmse_mean','mae_mean','da_mean']].to_csv(HERE / 'table6_endpoint_errors.csv', index=False)
    paired.query('experiment == "peer_correction"').to_csv(HERE / 'table5_paired_intervals.csv', index=False)
    online_ci = paired.query('experiment == "online_updates"')
    online_ci.to_csv(HERE / 'tables7_8_update_intervals.csv', index=False)
    seeds.to_csv(HERE / 'all_by_seed.csv', index=False)
    daily.to_csv(HERE / 'all_daily.csv.gz', index=False)

    curve = daily.query('experiment == "online_updates"').sort_values(count_keys + ['as_of_date']).copy()
    traces = curve.groupby(count_keys, sort=False)
    curve['session'] = traces.cumcount() + 1
    curve['annualized_return'] = curve.nav ** (252 / curve.session) - 1
    curve = (curve.loc[curve.session.ge(60)].groupby(['market','model','as_of_date','session'], as_index=False)
             .annualized_return.mean())
    curve.to_csv(HERE / 'figure3_annualized_curves.csv', index=False)
    online_ci.query('metric == "annualized_return_pp" and treatment == @MODELS[-1] and control == @MODELS[0] and block_days == 10'
                    ).to_csv(HERE / 'figure3_intervals.csv', index=False)

    lines = ['# 三市场复权消融结果', '',
             '本汇总对应论文表 4–8 和图 3。各市场沿用已完成的复权重训检查点、固定股票名单、测试样本和冻结 Kronos 特征。组合每日以次日开盘成交、收盘估值，按 15 个基点逐边扣费，并按前次复权实验的规则计入分红送转。', '',
             '## 表 4 在线更新消融', '',
             '括号内为三个随机种子的样本标准差。', '']
    data = online.set_index(['market','model'])
    rows = []
    for market in MARKETS:
        for name in MODELS:
            r = data.loc[market, name]
            rows.append([TITLES[market], MODEL_ZH[name], f'{r.annualized_return_mean:.2%} ({r.annualized_return_std:.2%})',
                         f'{r.rankic_mean:.5f} ({r.rankic_std:.5f})', f'{r.sharpe_mean:.3f}'])
    lines += markdown_table(['市场','策略','扣费年化收益','RankIC','Sharpe'], rows)
    lines += ['', '## 表 5 跨股票修正消融', '',
              '四种配置共享同一条联合更新模型轨迹。收益和 RankIC 均为三个种子的平均。',
              '']
    data = peer.set_index(['market','model'])
    rows = []
    for market in MARKETS:
        for name in VARIANTS:
            r = data.loc[market, name]
            rows.append([TITLES[market], PEER_ZH[name], f'{r.annualized_return_mean:.2%}', f'{r.rankic_mean:.5f}'])
    lines += markdown_table(['市场','配置','扣费年化收益','RankIC'], rows)
    lines += ['', '在线 PCA 相对各配置的扣费年化收益差。95% 配对区间使用 10 日段抽样，单位为百分点。', '']
    data = paired.query('experiment == "peer_correction" and metric == "annualized_return_pp"').set_index(['market','control'])
    rows = []
    for market in MARKETS:
        for name in VARIANTS[:3]:
            r = data.loc[market, name]
            low = 0. if abs(r.low) < 1e-10 else r.low
            rows.append([TITLES[market], PEER_ZH[name], f'{r.delta:+.2f}', f'[{low:.2f}, {r.high:.2f}]'])
    lines += markdown_table(['市场','比较基准','差值','95% 配对区间'], rows)
    lines += ['', '## 表 6 第 10 日预测误差', '',
              '沿用论文的第 10 日预测目标和零收益符号判定方式，并补齐纳斯达克 100。',
              '']
    data = online.set_index(['market','model'])
    rows = []
    for market in MARKETS:
        for name in MODELS:
            r = data.loc[market, name]
            rows.append([TITLES[market], MODEL_ZH[name], f'{r.rmse_mean:.5f}', f'{r.mae_mean:.5f}', f'{r.da_mean:.2%}'])
    lines += markdown_table(['市场','策略','RMSE','MAE','方向准确率'], rows)
    lines += ['', '## 表 7–8 在线策略差值', '',
              '这里列出扣费年化收益的百分点差值。CSV 同时包含 RankIC 的配对区间。两种指标都使用各策略共用的日期块和 5,000 次抽样。',
              '']
    data = online_ci.query('metric == "annualized_return_pp"').set_index(['market','treatment','control','block_days'])
    rows = []
    for market in MARKETS:
        for treatment, control in online_pairs:
            values = [data.loc[market,treatment,control,size] for size in (10,20,40)]
            rows.append([TITLES[market], f'{MODEL_ZH[treatment]} vs. {MODEL_ZH[control]}',
                         f'{values[0].delta:+.2f}', *[f'[{r.low:.2f}, {r.high:.2f}]' for r in values]])
    lines += markdown_table(['市场','比较','差值','10 日块','20 日块','40 日块'], rows)
    lines += ['', '## 文件', '',
              '- [表 4 在线更新消融](table4_online_updates.csv)',
              '- [表 5 跨股票消融](table5_peer_correction.csv) · [配对区间](table5_paired_intervals.csv)',
              '- [表 6 预测误差](table6_endpoint_errors.csv) · [表 7–8 配对区间](tables7_8_update_intervals.csv)',
              '- [图 3 曲线数据](figure3_annualized_curves.csv) · [区间数据](figure3_intervals.csv)',
              '- [逐种子指标](all_by_seed.csv) · [每日收益和 RankIC](all_daily.csv.gz)',
              '',
              '年化收益先对各随机种子按 252 个交易日计算，再对种子求平均。曲线从第 60 个交易日开始展示，净值从首笔交易累计。表 4 三种策略的逐日净收益和 RankIC 与此前复权结果逐点一致。表 5 的在线 PCA 预测与表 4 的完整模型逐点一致。',
              '',
              'CSI300 沿用 2026 年 8 月 20 日成分股名单。这些消融反映该固定名单内的方法差异。美股分红按除息日入账，标普 500 的 APTV 分拆按现金等值处理。']
    (HERE / '复权消融结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Verified {len(seeds)} seed rows, {len(daily)} daily rows, {len(paired)} paired intervals.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-archive', type=Path, metavar='DIRECTORY')
    options = parser.parse_args()
    if options.verify_archive is None:
        main()
    else:
        verify_archive(options.verify_archive)
