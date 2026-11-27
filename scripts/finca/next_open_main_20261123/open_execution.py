"""Daily open execution and close valuation for target portfolios."""
import numpy as np
import pandas as pd

COST = .0015

def rebalance(position, cash, price, tradable, desired):
    """Solve the post-fee value that funds the day's target portfolio."""
    old = position * price
    capital = old.sum() + cash
    budget = capital - old[~tradable].sum()
    target_weight = np.where(tradable, desired, 0.)
    assert target_weight.sum() <= 1 + 1e-12
    def fees(value):
        return COST * np.abs(target_weight * value - np.where(tradable, old, 0.)).sum()
    lo, hi = 0., budget
    assert budget >= 0 and fees(lo) <= budget + 1e-12
    for _ in range(45):
        mid = (lo + hi) / 2.
        if mid + fees(mid) <= budget:
            lo = mid
        else:
            hi = mid
    new = np.where(tradable, target_weight * lo, old)
    turnover = np.abs(new - old).sum()
    fee = COST * turnover
    cash = capital - new.sum() - fee
    assert cash >= -1e-10
    units = np.where(tradable, new / price, position)
    np.testing.assert_allclose((units * price).sum() + cash + fee, capital, atol=1e-12, rtol=0)
    return units, max(cash, 0.), turnover / capital, fee / capital

def run_open(targets, calendar, opens, closes):
    symbols = opens.columns
    units = np.zeros(len(symbols))
    cash = 1.
    previous_nav = 1.
    last_close = closes.iloc[0].fillna(1.).to_numpy(float)
    rows = []
    for origin, execution_date in calendar.items():
        desired = targets.loc[origin].to_numpy(float)
        opening = opens.loc[execution_date].to_numpy(float)
        closing = closes.loc[execution_date].to_numpy(float)
        tradable = np.isfinite(opening)
        marks_at_open = np.where(tradable, opening, last_close)
        missing_orders = (~tradable) & ((desired > 0) | (units > 0))
        units, cash, turnover, fee = rebalance(units, cash, marks_at_open, tradable, desired)
        closing_marks = np.where(np.isfinite(closing), closing, marks_at_open)
        nav = cash + units @ closing_marks
        assert nav > 0 and np.isfinite(nav)
        rows.append(dict(as_of_date=origin,execution_date=execution_date,net_return=nav/previous_nav-1,
                         nav=nav,turnover=turnover,fee_rate=fee,missing_open_positions=missing_orders.sum(),
                         locked_positions=((~tradable)&(units>0)).sum()))
        previous_nav = nav
        last_close = closing_marks
    return pd.DataFrame(rows)
