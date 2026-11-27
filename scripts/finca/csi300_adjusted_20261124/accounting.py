"""Price adjustment and a next-open account with explicit corporate actions."""
import numpy as np
import pandas as pd

from scripts.finca.next_open_main_20261123.open_execution import rebalance


def adjustment_ratios(dates, factors):
    """Normalize cumulative factors to the last observed date."""
    selected = factors.sort_index().reindex(dates, method='ffill')
    if selected.isna().any() or (selected <= 0).any():
        raise ValueError('missing or invalid corporate-action factor')
    return selected / selected.iloc[-1]


def apply_actions(units, cash, last_close, dividend, share_multiplier):
    cash += float(units @ dividend)
    return units * share_multiplier, cash, (last_close - dividend) / share_multiplier


def run_open(targets, calendar, opens, closes, dividends, multipliers):
    units = np.zeros(len(opens.columns))
    cash = previous_nav = 1.
    last_close = closes.iloc[0].fillna(1.).to_numpy(float)
    rows = []
    for origin, date in calendar.items():
        opening = opens.loc[date].to_numpy(float)
        closing = closes.loc[date].to_numpy(float)
        dividend = dividends.loc[date].to_numpy(float)
        multiplier = multipliers.loc[date].to_numpy(float)
        payout = float(units @ dividend)
        units, cash, last_close = apply_actions(units, cash, last_close, dividend, multiplier)
        tradable = np.isfinite(opening)
        open_marks = np.where(tradable, opening, last_close)
        units, cash, turnover, fee = rebalance(
            units, cash, open_marks, tradable, targets.loc[origin].to_numpy(float))
        last_close = np.where(np.isfinite(closing), closing, open_marks)
        nav = float(cash + units @ last_close)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f'invalid NAV at {date}')
        rows.append(dict(as_of_date=origin, execution_date=date, net_return=nav / previous_nav - 1,
                         nav=nav, turnover=turnover, fee_rate=fee, cash_dividend=payout))
        previous_nav = nav
    return pd.DataFrame(rows)
