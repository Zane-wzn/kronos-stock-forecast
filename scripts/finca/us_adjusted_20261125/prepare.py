"""Align US corporate actions, adjusted targets and execution quotes to paper panels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA, ROOT, RUNS, unique_path

HERE = RUNS / 'us_adjusted_20261125'
from scripts.finca.next_open_main_20261123.replay_markets import load_panel, RAW

KEYS = ['as_of_date', 'stock']
SPLITS = ('train', 'val', 'test')
EXCLUDED_SPLITS = {
    'CVNA': {'2026-05-06','2026-05-07'},
    'DD': {'2025-11-03'},
    'HON': {'2025-10-30'},
    'GE': {'2023-01-04','2024-04-02'},
    'DHR': {'2023-10-02'},
    'IBM': {'2021-11-04'},
}


def split_scale(dates, events):
    values = pd.Series(1., index=dates)
    for day, ratio in events.items():
        values.loc[values.index < day] *= ratio
    return values


def cash_adjusted_close(close, dividends):
    previous = close.ffill().shift()
    cash = pd.Series(dividends, dtype=float).reindex(close.index, fill_value=0.)
    divisor = 1 - (cash / previous).where(cash.ne(0), 0.)
    if (divisor.dropna() <= 0).any():
        raise ValueError('cash distribution is at least the preceding close')
    return close / divisor.fillna(1.).cumprod()


def label_prices(matrix, dates, stocks):
    dates = np.asarray(dates)
    row = matrix.index.get_indexer(dates.ravel()).reshape(dates.shape)
    col = matrix.columns.get_indexer(stocks)
    if (row < 0).any() or (col < 0).any():
        raise ValueError('date or stock absent from price grid')
    return matrix.to_numpy()[row, col[:, None] if row.ndim == 2 else col]


def source(name, symbol):
    path = HERE / 'data' / name / f'{symbol}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def grouped(rows):
    values = {}
    for day, value in set(map(tuple, rows)):
        values[day] = values.get(day, 0.) + value
    return values


def action_inputs(stock, frame, nasdaq_raw):
    symbol = stock.removeprefix('us.')
    preferred = source('splithistory', symbol)
    alternate = source('split', symbol)
    splits = dict(alternate or [])
    splits.update(preferred or [])
    splits = {day: ratio for day, ratio in splits.items() if '2020-01-01' <= day <= '2026-08-19'
              and ratio != 1 and day not in EXCLUDED_SPLITS.get(symbol, ())}
    scale = split_scale(frame.index, splits)
    direct = source('sa_div', symbol)
    direct_cash = grouped((row['dt'], float(row['amt'])) for row in direct or [] if row.get('amt'))
    nasdaq = source('nasdaq', symbol)
    nd_rows = []
    for row in nasdaq or []:
        date = pd.to_datetime(row['exOrEffDate'], errors='coerce')
        if pd.notna(date) and row.get('currency') == 'USD' and row['amount'].startswith('$'):
            nd_rows.append([date.date().isoformat(), float(row['amount'][1:].replace(',', ''))])
    nd_cash = grouped(nd_rows)
    # Where available use observed adjustments to check the effective dates and
    # whether vendor cash amounts are in historical or split-normalized units.
    provider = source('price', symbol)
    exact = None
    source_close = frame.close
    if symbol in nasdaq_raw:
        exact = nasdaq_raw[symbol]['adj_close'].reindex(frame.index) / nasdaq_raw[symbol].close.reindex(frame.index)
    elif provider:
        prices = pd.DataFrame(provider, columns=['date', 'close', 'adj']).set_index('date')
        exact = (prices.adj / prices.close).reindex(frame.index)
        source_close = prices.close.reindex(frame.index)
    previous = frame.close.shift()
    implied = (1 - exact.shift() / exact) * previous if exact is not None else pd.Series(dtype=float)
    relative_scale = source_close / frame.close
    scale_change = relative_scale.div(relative_scale.shift())
    events = {}
    for day in sorted(set(direct_cash) | set(nd_cash)):
        if day not in frame.index or not '2020-01-01' <= day <= '2026-06-22':
            continue
        a, n = direct_cash.get(day), nd_cash.get(day)
        factor = scale.loc[day]
        estimate = implied.get(day, np.nan)
        if symbol == 'AZN' and np.isfinite(estimate) and abs(estimate) < .01:
            neighbors = implied.loc[:day].tail(3).tolist() + implied.loc[day:].head(3).tolist()
            if any(np.isfinite(value) and value > .1 for value in neighbors):
                continue
        candidates = [v for v in (a, n / factor if n else None) if v]
        if not candidates:
            continue
        if np.isfinite(estimate) and estimate > 0:
            events[day] = min(candidates, key=lambda v: abs(v - estimate))
        else:
            events[day] = candidates[0]
    # Infer a missing cash distribution only from a discontinuity in an
    # independently published adjusted-close ratio (same split basis).
    inferred = []
    for day, value in implied.items():
        if '2020-01-01' <= day <= '2026-06-22' and day not in events and value >= .008:
            # Price bases can already embed a spin-off. The factor jump from
            # the other vendor would count that distribution twice.
            if np.isfinite(scale_change.get(day, np.nan)) and abs(scale_change[day] - 1.) > .02:
                continue
            movement = value / previous.loc[day]
            if movement > 1e-4:
                events[day] = value
                inferred.append(day)
    return scale, splits, events, inferred, (direct is not None, bool(nasdaq), exact is not None, preferred is not None or alternate is not None)


def main(market):
    out = HERE / market
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / f'.peer_pca_inputs/{market}/hidden_cache'
    folder = unique_path(DATA, '*/' + RAW[market])
    nasdaq_dir = unique_path(DATA, '*/' + RAW['nasdaq100'])
    nasdaq_raw = {file.stem.removeprefix('us_'): pd.read_csv(file).set_index('date') for file in nasdaq_dir.glob('us_*.csv')}
    panels = {s: pd.read_csv(cache / f'meta_{s}.csv', usecols=lambda c: c in KEYS + ['label_dates','label_start_date','label_end_date']) for s in SPLITS}
    original, full, calendar = load_panel(market)
    unique = sorted(set(pd.concat([panel.stock for panel in panels.values()])))
    extra = full.iloc[len(original):]
    unique = sorted(set(unique) | set(extra.stock))
    matrix, adjusted, opens, closes, distributions, multipliers = {}, {}, {}, {}, {}, {}
    audit, actions = [], []
    for stock in unique:
        symbol = stock.removeprefix('us.')
        path = folder / f'us_{symbol}.csv'
        if not path.exists():
            path = folder / f'us_{symbol.replace("-", "_")}.csv'
        frame = pd.read_csv(path, usecols=['date','open','close']).set_index('date').sort_index()
        scale, splits, cash, inferred, coverage = action_inputs(stock, frame, nasdaq_raw)
        if market == 'nasdaq100':
            adjusted[stock] = nasdaq_raw[symbol].adj_close
        else:
            adjusted[stock] = cash_adjusted_close(frame.close, cash)
        matrix[stock] = frame.close
        opens[stock], closes[stock] = frame.open * scale, frame.close * scale
        pay = pd.Series(cash, dtype=float) * scale
        split = pd.Series(splits, dtype=float)
        distributions[stock], multipliers[stock] = pay, split
        for day in sorted(set(cash) | set(splits)):
            actions.append(dict(stock=stock,date=day,dividend=cash.get(day,0.) * scale.get(day,1.),
                                split=splits.get(day,1.),inferred=day in inferred))
        audit.append(dict(stock=stock,stockanalysis_dividends=coverage[0],nasdaq_dividends=coverage[1],
                          independent_adj_close=coverage[2],split_history=coverage[3],
                          cash_events=len(cash),inferred_cash=len(inferred),splits=len(splits)))
    raw, adj = pd.DataFrame(matrix).sort_index(), pd.DataFrame(adjusted).sort_index()
    if market == 'nasdaq100':
        for panel in panels.values():
            offsets = raw.index.get_indexer(panel.label_start_date)
            assert np.all(offsets >= 0)
            target_dates = raw.index.to_numpy()[offsets[:, None] + np.arange(10)]
            np.testing.assert_array_equal(target_dates[:, -1], panel.label_end_date.to_numpy())
            panel['label_dates'] = ['|'.join(row) for row in target_dates]
    if not extra.empty:
        mapped = panels['test'].drop_duplicates('as_of_date').set_index('as_of_date').label_dates
        panels['extra'] = extra[KEYS].assign(label_dates=extra.as_of_date.map(mapped))
    supplement = (list(DATA.rglob('hidden_val.npy')))
    key = 'paper_suite_20261112' if market == 'nasdaq100' else '\u5171\u4eab\u8d44\u6e90'
    backup, = [p.parent for p in supplement if market in p.parts and key in p.parts]
    statistics = {}
    for name, panel in panels.items():
        targets = panel.label_dates.str.split('|', expand=True).to_numpy()
        expected_path = (cache if name == 'test' else backup) / f'labels_{name}.npy'
        if name == 'extra':
            with np.load(HERE.parent / 'next_open_main_20261123/dasfaa_extra_features_sp500.npz') as f:
                expected = f['labels']
        else:
            expected = np.load(expected_path)
        raw_return = label_prices(raw, targets, panel.stock) / label_prices(raw, panel.as_of_date, panel.stock)[:, None] - 1.
        take = np.isfinite(expected) & np.isfinite(raw_return)
        np.testing.assert_allclose(raw_return[take], expected[take], atol=2e-6, rtol=0)
        new = label_prices(adj, targets, panel.stock) / label_prices(adj, panel.as_of_date, panel.stock)[:, None] - 1.
        new[~np.isfinite(expected)] = np.nan
        if not np.array_equal(np.isfinite(new), np.isfinite(expected)):
            raise ValueError(f'{market} {name}: adjusted observations differ from the original label coverage')
        computed = new.astype('float32')
        target = out / f'labels_{name}.npy'
        changed_from_previous = not target.exists() or not np.array_equal(np.load(target), computed, equal_nan=True)
        if changed_from_previous:
            np.save(target, computed)
        print(market, name, 'label file changed:', changed_from_previous, flush=True)
        statistics[name] = dict(rows=len(panel), changed=int(np.count_nonzero(np.abs(new[take] - expected[take]) > 1e-6)))
    adj.to_csv(out / 'adjusted_close.csv.gz')
    dates = pd.Index([calendar.index[0], *calendar], name='date')
    for name, values, fill in [('opens', opens, np.nan), ('closes', closes, np.nan),
                               ('dividends', distributions, 0.), ('splits', multipliers, 1.)]:
        data = pd.DataFrame(values).reindex(index=dates, columns=sorted(full.stock.unique()))
        data.fillna(fill).to_csv(out / f'{name}.csv')
    pd.DataFrame(audit).to_csv(out / 'action_coverage.csv', index=False)
    pd.DataFrame(actions).sort_values(['date','stock']).to_csv(out / 'actions.csv', index=False)
    (out / 'data_audit.json').write_text(json.dumps(statistics, indent=2), encoding='utf-8')
    print(market, statistics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('market', choices=['nasdaq100','sp500'])
    main(parser.parse_args().market)
