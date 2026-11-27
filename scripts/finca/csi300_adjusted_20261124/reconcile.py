"""Match test-period factor changes to public cash and share distributions."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from .fetch import fetch
from .prepare import HERE, matrices, phased_events


def read_sina(stock):
    code = stock.split('.')[1]

    def parse(response):
        html = BeautifulSoup(response.content.decode('gb18030', errors='replace'), 'html.parser')
        table = html.select_one('#sharebonus_1')
        if table is None:
            raise ValueError(f'no distribution table for {stock}')
        rows = []
        for row in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all('td', recursive=False)]
            if len(cells) == 9 and re.fullmatch(r'\d{4}-\d{2}-\d{2}', cells[5]) and cells[4] == '实施':
                rows.append(dict(stock=stock, date=cells[5], record_date=cells[6],
                                 dividend=float(cells[3]) / 10.,
                                 share_bonus=(float(cells[1]) + float(cells[2])) / 10.))
        return rows

    url = f'https://money.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml'
    return fetch(HERE / f'data/sina_events_{stock}.json', url, parse)


def main():
    quotes, factor, history = matrices()
    actions = phased_events(history).assign(source='Eastmoney')
    dates = ('2025-07-02', '2026-06-08')
    moves = factor.rename_axis(columns='stock').div(factor.shift()).stack().rename('factor_change')
    moves = moves[~np.isclose(moves, 1., rtol=0, atol=1e-10)].reset_index()
    moves = moves[moves.date.between(*dates)]
    keys = ['date', 'stock']
    gaps = moves.merge(actions[keys], on=keys, how='left', indicator=True).query("_merge == 'left_only'")
    if len(gaps):
        with ThreadPoolExecutor(max_workers=6) as executor:
            records = [record for rows in executor.map(read_sina, sorted(gaps.stock.unique())) for record in rows]
        supplemental = (gaps[keys].merge(pd.DataFrame(records), on=keys, validate='one_to_many')
                        .groupby(keys, as_index=False)[['dividend', 'share_bonus']].sum()
                        .assign(source='Sina'))
        actions = pd.concat([actions, supplemental], ignore_index=True).sort_values(keys)
    test = actions[actions.date.between(*dates)]
    comparison = moves.merge(test, on=keys, how='outer', indicator=True, validate='one_to_one')
    previous = quotes.close.ffill().shift().stack().reindex(comparison.set_index(keys).index).to_numpy()
    comparison['reference_difference'] = (previous / comparison.factor_change
                                           - (previous - comparison.dividend) / (1 + comparison.share_bonus))
    comparison.to_csv(HERE / 'factor_event_crosscheck.csv', index=False)
    if not comparison['_merge'].eq('both').all():
        raise ValueError(f'unmatched actions: {comparison[comparison["_merge"].ne("both")]}')
    actions.to_csv(HERE / 'corporate_actions.csv', index=False)
    audit = json.loads((HERE / 'data_audit.json').read_text(encoding='utf-8'))
    audit.update(action_rows=len(actions), test_actions=len(test),
                 supplemental_test_actions=int(test.source.eq('Sina').sum()),
                 test_theoretical_price_differences_over_one_cent=int(comparison.reference_difference.abs().gt(.01001).sum()))
    (HERE / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'{len(test)} test-period actions matched to factor changes')
    print(test.source.value_counts().to_string())
    print(comparison.loc[comparison.reference_difference.abs().gt(.01001)].to_string(index=False))


if __name__ == '__main__':
    main()
