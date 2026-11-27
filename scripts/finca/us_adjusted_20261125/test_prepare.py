import unittest
import numpy as np
import pandas as pd
from scripts.finca.us_adjusted_20261125.prepare import cash_adjusted_close, split_scale, label_prices


class AdjustmentTest(unittest.TestCase):
    def test_split_scale_excludes_effective_day(self):
        dates = pd.Index(['2025-01-02','2025-01-03','2025-01-06'])
        self.assertEqual(split_scale(dates, {'2025-01-03': 2., '2025-01-06': 3.}).tolist(), [6., 3., 1.])

    def test_cash_adjustment_keeps_return_flat_with_no_price_move(self):
        close = pd.Series([10., 9., 9.], index=['2025-01-02','2025-01-03','2025-01-06'])
        base = cash_adjusted_close(close, {'2025-01-03': 1.})
        np.testing.assert_allclose(base, [10., 10., 10.])

    def test_target_date_preserves_missing_quote(self):
        quote = pd.DataFrame({'A': [10.,np.nan,12.]}, index=['2025-01-02','2025-01-03','2025-01-06'])
        result = label_prices(quote, [['2025-01-03','2025-01-06']], ['A'])
        np.testing.assert_allclose(result, [[np.nan,12.]], equal_nan=True)
