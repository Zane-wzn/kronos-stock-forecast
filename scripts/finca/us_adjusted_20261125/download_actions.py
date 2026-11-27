"""Download public US distribution, split, and adjusted price histories."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from threading import local
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

from ..paths import ROOT, RUNS

HERE = RUNS / 'us_adjusted_20261125'
THREAD = local()
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36'}
SOURCES = ('sa_div', 'splithistory')


def yahoo_symbol(stock):
    return stock.removeprefix('us.').replace('.', '-')


def get(url, referer):
    if not hasattr(THREAD, 'session'):
        THREAD.session = requests.Session()
    return THREAD.session.get(url, headers={**HEADERS, 'Referer': referer}, timeout=12)


def split_history(symbol):
    url = f'https://stockscan.io/stocks/{symbol.replace("-", ".")}/stock-split'
    response = get(url, 'https://stockscan.io/')
    if response.status_code == 404:
        return None
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    if not soup.title or 'StockScan' not in soup.title.get_text():
        raise ValueError('invalid split page')
    for table in soup.find_all('table'):
        header = [cell.get_text(' ', strip=True) for cell in table.select('th')]
        if header[:2] == ['Date', 'Split Ratio']:
            rows = []
            for row in table.select('tbody tr'):
                date, ratio = [cell.get_text(' ', strip=True) for cell in row.select('td')][:2]
                left, right = re.split(r'\s*-\s*for\s*-\s*', ratio, flags=re.I)
                rows.append([date, float(left) / float(right)])
            return rows
    return []


def dividends(symbol):
    url = f'https://api.nasdaq.com/api/quote/{symbol.replace("-", ".")}/dividends?assetclass=stocks'
    response = get(url, 'https://www.nasdaq.com/')
    response.raise_for_status()
    payload = response.json()
    if payload['status']['rCode'] != 200:
        raise ValueError(payload['status'])
    rows = ((payload.get('data') or {}).get('dividends') or {}).get('rows')
    return rows or []


def prices(symbol):
    url = f'https://api.stockanalysis.com/api/symbol/s/{symbol.replace("-", ".")}/history?range=5Y'
    response = get(url, 'https://stockanalysis.com/')
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if payload['status'] != 200:
        raise ValueError(payload['status'])
    return [[row['t'], row['c'], row.get('a', row['c'])] for row in payload['data']]


def stockanalysis_dividends(symbol):
    url = f'https://api.stockanalysis.com/api/symbol/s/{symbol.replace("-", ".")}/dividend?chart=history&range=10Y'
    response = get(url, 'https://stockanalysis.com/')
    response.raise_for_status()
    payload = response.json()
    if payload['status'] != 200 or not isinstance(payload['data'], list):
        raise ValueError(payload['status'])
    return payload['data']


def dividend_history(symbol):
    name = symbol.replace('-', '.')
    response = get(f'https://www.dividendhistory.org/payout/{name}/', 'https://www.dividendhistory.org/')
    if response.status_code == 404:
        return None
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    if not soup.title or 'Dividend History' not in soup.title.get_text():
        raise ValueError('invalid distribution page')
    rows = []
    for row in soup.select('tr'):
        cells = [cell.get_text(' ', strip=True) for cell in row.find_all('td', recursive=False)]
        if len(cells) >= 3 and re.fullmatch(r'\d{4}-\d{2}-\d{2}', cells[0]) and cells[2].startswith('$'):
            if 'estimated' not in ' '.join(cells[3:]).lower():
                rows.append([cells[0], float(cells[2][1:].replace(',', ''))])
    return rows


def splits_alternative(symbol):
    options = [symbol.lower(), symbol.lower().replace('-', '.'), symbol.lower().replace('-', '')]
    for name in dict.fromkeys(options):
        response = get(f'https://www.splithistory.com/{name}/', 'https://www.splithistory.com/')
        if response.status_code == 404:
            continue
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        if not soup.title or 'Split History' not in soup.title.get_text():
            raise ValueError('invalid split page')
        rows = []
        for day, left, right in re.findall(r'(\d{1,2}/\d{1,2}/\d{4})\s+([\d.]+)\s+for\s+([\d.]+)',
                                           soup.get_text(' ', strip=True), re.I):
            rows.append([pd.to_datetime(day).date().isoformat(), float(left) / float(right)])
        return rows
    return None


def worker(task):
    source, symbol = task
    path = HERE / 'data' / source / f'{symbol}.json'
    if path.exists():
        return
    for attempt in range(3):
        try:
            data = {'nasdaq': dividends, 'split': split_history, 'price': prices,
                    'divhistory': dividend_history, 'splithistory': splits_alternative,
                    'sa_div': stockanalysis_dividends}[source](symbol)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(data), encoding='utf-8')
            temp.replace(path)
            return
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            if attempt == 2:
                return source, symbol, str(error)
            time.sleep(attempt + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markets', nargs='+', default=['nasdaq100', 'sp500'])
    parser.add_argument('--sources', nargs='+', default=list(SOURCES))
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    for source in args.sources:
        (HERE / 'data' / source).mkdir(parents=True, exist_ok=True)
    symbols = sorted(set().union(*(pd.read_csv(ROOT / f'.peer_pca_inputs/{market}/hidden_cache/meta_{split}.csv',
                                               usecols=['stock']).stock.map(yahoo_symbol)
                                  for market in args.markets for split in ['train', 'val', 'test'])))
    tasks = [(source, symbol) for source in args.sources for symbol in (args.symbols or symbols)]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(worker, task) for task in tasks]
        for i, future in enumerate(as_completed(pending), 1):
            if result := future.result():
                failures.append(result)
            if i % 100 == 0 or i == len(tasks):
                print(f'{i}/{len(tasks)} source requests, {len(failures)} failures', flush=True)
    (HERE / ('download_failures_' + '_'.join(args.sources) + '.json')).write_text(
        json.dumps(failures, indent=2), encoding='utf-8')
    print(failures[:12], flush=True)
