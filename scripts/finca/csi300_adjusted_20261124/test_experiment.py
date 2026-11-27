import unittest

import numpy as np
import pandas as pd

from scripts.finca.csi300_adjusted_20261124.accounting import adjustment_ratios, apply_actions, run_open


class CorporateActionTest(unittest.TestCase):
    def test_factor_used_on_effective_date(self):
        dates = pd.Index(['2025-01-02', '2025-01-03', '2025-01-06'])
        factors = pd.Series([1.0, 2.0, 4.0], index=['2000-01-01', '2025-01-03', '2025-01-07'])
        np.testing.assert_allclose(adjustment_ratios(dates, factors), [.5, 1., 1.])

    def test_dividend_goes_to_existing_holders_before_open(self):
        units, cash, marks = apply_actions(np.array([2., 0.]), 1., np.array([10., 10.]),
                                           np.array([1., 1.]), np.array([2., 1.]))
        np.testing.assert_allclose(units, [4., 0.])
        np.testing.assert_allclose(marks, [4.5, 9.])
        self.assertEqual(cash, 3.)

    def test_cash_and_split_offset_the_ex_date_price_drop(self):
        dates = ['2025-01-02', '2025-01-03', '2025-01-06']
        calendar = pd.Series(dates[1:], index=dates[:-1])
        names = ['old', 'new']
        quotes = pd.DataFrame([[10., 20.], [10., 20.], [4.5, 10.]], index=dates, columns=names)
        targets = pd.DataFrame([[1., 0.], [0., 1.]], index=dates[:-1], columns=names)
        payouts = pd.DataFrame([[0., 0.], [0., 0.], [1., 0.]], index=dates, columns=names)
        splits = pd.DataFrame([[1., 1.], [1., 1.], [2., 2.]], index=dates, columns=names)
        daily = run_open(targets, calendar, quotes, quotes, payouts, splits)
        self.assertGreater(daily.nav.iloc[1], .99)
        self.assertAlmostEqual(daily.cash_dividend.iloc[1], daily.nav.iloc[0] / 10.)


if __name__ == '__main__':
    unittest.main()
