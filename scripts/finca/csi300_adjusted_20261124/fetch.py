"""Cache public Sina cumulative factors and Eastmoney corporate actions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import pandas as pd
import requests

from ..paths import ROOT, RUNS

HERE = RUNS / 'csi300_adjusted_20261124'
DATA = HERE / 'data'
EAST = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
SINA = 'https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js'


def fetch(path, url, parse, params=None):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = parse(response)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            temporary.replace(path)
            return data
        except (requests.RequestException, ValueError, KeyError) as error:
            if attempt == 3:
                raise RuntimeError(f'{path.name}: {error}') from error
            time.sleep(.5 * (attempt + 1))


def fetch_factor(symbol):
    def parse(response):
        payload, _ = json.JSONDecoder().raw_decode(response.text.split('=', 1)[1].lstrip())
        data = payload['data']
        if not data:
            raise ValueError('empty adjustment history')
        return data

    return fetch(DATA / f'{symbol}.json', SINA.format(symbol=symbol.replace('.', '')), parse)


def fetch_events(item):
    number, symbols = item
    codes = ','.join(json.dumps(symbol.split('.')[1]) for symbol in symbols)
    params = dict(reportName='RPT_SHAREBONUS_DET', columns='ALL', pageSize=500,
                  sortColumns='EX_DIVIDEND_DATE,SECUCODE', sortTypes='1,1',
                  filter=f"(EX_DIVIDEND_DATE>='2016-01-01')(EX_DIVIDEND_DATE<='2026-06-30')(SECURITY_CODE in ({codes}))")

    def parse(response):
        payload = response.json()['result']
        if payload is None:
            raise ValueError('missing dataset')
        return payload

    rows = []
    for page in range(1, 20):
        payload = fetch(DATA / f'events_{number}_{page}.json', EAST, parse, {**params, 'pageNumber': page})
        rows += payload['data']
        if page == payload['pages']:
            if len(rows) != payload['count']:
                raise ValueError(f'incomplete event group {number}')
            return rows
    raise ValueError(f'too many pages for group {number}')


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    cache = ROOT / '.peer_pca_inputs/csi300/hidden_cache'
    stocks = sorted(set().union(*(pd.read_csv(cache / f'meta_{split}.csv', usecols=['stock']).stock
                                  for split in ['train', 'val', 'test'])))
    groups = list(enumerate(stocks[start:start + 35] for start in range(0, len(stocks), 35)))
    with ThreadPoolExecutor(max_workers=10) as executor:
        factor_downloads = [executor.submit(fetch_factor, stock) for stock in stocks]
        event_downloads = [executor.submit(fetch_events, group) for group in groups]
        events = [row for task in event_downloads for row in task.result()]
        failures = []
        for stock, task in zip(stocks, factor_downloads):
            try:
                task.result()
            except RuntimeError as error:
                failures.append((stock, str(error)))
    print(f'factors {len(stocks) - len(failures)}/{len(stocks)}, Eastmoney rows {len(events)}', flush=True)
    if failures:
        print(failures, flush=True)
        raise SystemExit(1)
