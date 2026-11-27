"""Build factor-adjusted labels on exactly the archived CSI300 panels."""
import json

import numpy as np
import pandas as pd

from .accounting import adjustment_ratios
from ..paths import DATA, ROOT, RUNS, unique_path

HERE = RUNS / 'csi300_adjusted_20261124'
ARCHIVE = RUNS / 'csi300_next_open_20261123'
CACHE = ROOT / '.peer_pca_inputs/csi300/hidden_cache'
SPLITS = ('train', 'val', 'test')
KEYS = ['as_of_date', 'stock']


def matrices():
    raw = unique_path(DATA, '*/csi300_daily/csi300_daily')
    quotes, factors = {}, {}
    for path in sorted(raw.glob('*.csv')):
        stock = path.stem.replace('_', '.')
        quotes[stock] = pd.read_csv(path, usecols=['date', 'open', 'close']).set_index('date')
        data = pd.read_json(HERE / f'data/{stock}.json', dtype={'d': str, 'f': float}, convert_dates=False)
        factors[stock] = data.set_index('d').f
    quotes = pd.concat(quotes, names=['stock']).unstack('stock').sort_index()
    dates = quotes.index
    changes = pd.DataFrame({stock: adjustment_ratios(dates, factor) for stock, factor in factors.items()}, index=dates)
    return quotes, changes, factors


def phased_events(factors):
    payload = [row for path in sorted((HERE / 'data').glob('events_*.json'))
               for row in json.loads(path.read_text(encoding='utf-8'))['data']]
    keys = ['stock', 'date']
    rows = []
    for item in payload:
        code, exchange = item['SECUCODE'].lower().split('.')
        stock = f'{exchange}.{code}'
        if stock in factors and item['EX_DIVIDEND_DATE']:
            rows.append(dict(stock=stock, date=item['EX_DIVIDEND_DATE'][:10], report=item['REPORT_DATE'],
                             dividend=(item['PRETAX_BONUS_RMB'] or 0) / 10.,
                             share_bonus=(item['BONUS_IT_RATIO'] or 0) / 10.))
    actions = pd.DataFrame(rows).drop_duplicates()
    if actions.duplicated(keys + ['report']).any():
        raise ValueError('conflicting actions for the same report and effective date')
    return (actions.groupby(keys, as_index=False)[['dividend', 'share_bonus']].sum()
            .sort_values(['date', 'stock']))


def values_at(matrix, dates, stocks):
    date_index = matrix.index.get_indexer(np.asarray(dates).ravel()).reshape(np.shape(dates))
    stock_index = matrix.columns.get_indexer(stocks)
    if (date_index < 0).any() or (stock_index < 0).any():
        raise ValueError('label date or symbol absent from the raw quote grid')
    stock_index = stock_index[:, None] if date_index.ndim == 2 else stock_index
    return matrix.to_numpy()[date_index, stock_index]


def labels_for(meta, close):
    targets = meta.label_dates.str.split('|', expand=True).to_numpy()
    origin = values_at(close, meta.as_of_date, meta.stock)
    return values_at(close, targets, meta.stock) / origin[:, None] - 1.


def main():
    supplement = unique_path(DATA, '*/csi300/*/reference/hidden_cache')
    quotes, ratios, factors = matrices()
    actions = phased_events(factors)
    actions.to_csv(HERE / 'corporate_actions.csv', index=False)
    close = quotes.close
    adjusted = close * ratios
    adjusted.to_csv(HERE / 'adjusted_close.csv.gz', compression='gzip')
    meta = {split: pd.read_csv(CACHE / f'meta_{split}.csv', usecols=KEYS + ['label_dates']) for split in SPLITS}
    calendar = meta['test'].drop_duplicates('as_of_date').set_index('as_of_date').label_dates
    with np.load(ARCHIVE / 'dasfaa_extra_features_csi300.npz') as extra:
        extra_meta = pd.DataFrame(dict(as_of_date=extra['dates'], stock=extra['stocks']))
        extra_meta['label_dates'] = extra_meta.as_of_date.map(calendar)
        meta['test_extra'] = extra_meta
        extra_labels = extra['labels']
    counts = {}
    for split, panel in meta.items():
        original = extra_labels if split == 'test_extra' else np.load(
            (CACHE if split == 'test' else supplement) / f'labels_{split}.npy')
        raw = labels_for(panel, close)
        corrected = labels_for(panel, adjusted)
        take = np.isfinite(raw) & np.isfinite(original)
        largest_raw_error = np.max(np.abs(raw[take] - original[take]))
        if largest_raw_error > 1e-6:
            raise ValueError(f'{split}: raw labels differ from archive by {largest_raw_error}')
        if split in ('train', 'val') and not np.isfinite(corrected).all():
            raise ValueError(f'{split}: expected complete labels')
        np.save(HERE / f'labels_{split}.npy', corrected.astype(np.float32))
        counts[split] = dict(rows=len(panel), finite_endpoints=int(np.isfinite(corrected[:, -1]).sum()),
                             changed_scalars=int(np.sum(np.abs(corrected[take] - raw[take]) > 1e-6)),
                             raw_label_max_error=float(largest_raw_error))
    report = dict(symbols=len(factors), action_rows=len(actions), action_symbols=actions.stock.nunique(),
                  test_actions=int(actions.date.between('2025-07-02', '2026-06-08').sum()), labels=counts)
    (HERE / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
