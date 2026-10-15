import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.experiments.run_dynamic_continual_csi300 import (
    build_dynamic_predictions,
    _load_and_join,
    _load_return_reference,
    rank_gaussian,
    update_hedge_weights,
    mix_normalized_signals,
    validate_initial_alpha,
)


class DynamicContinualTests(unittest.TestCase):
    def test_initial_point_three_mixing_and_weight_bounds(self):
        validate_initial_alpha(.3, .05, .95)
        np.testing.assert_allclose(mix_normalized_signals([1., -1.], [-1., 1.], .3), [.4, -.4])
        for bad in (.01, .99, np.nan):
            with self.assertRaises(ValueError):
                validate_initial_alpha(bad, .05, .95)
        self.assertEqual(update_hedge_weights(.7, .3, 0., 100.)['alpha'], .05)
        self.assertEqual(update_hedge_weights(.7, .3, 100., 0.)['alpha'], .95)

    def test_sixty_day_maturity_window_and_future_label_invariance(self):
        dates = pd.bdate_range('2025-07-01', '2025-11-04')
        rows = []
        for date in dates:
            for i in range(4):
                rows.append({'as_of_date': date, 'stock': str(i),
                    'static_raw_signal': float(i), 'continual_raw_signal': float(3-i),
                    'actual_signal': float(3-i), 'label_end_date': date + pd.Timedelta(days=14)})
        source = pd.DataFrame(rows)
        args = dict(test_start=str(dates.min().date()), test_end=str(dates.max().date()), initial_alpha=.3)
        a, log = build_dynamic_predictions(source, **args)
        self.assertTrue(log.matured_dates.le(60).all())
        self.assertEqual(log.iloc[-1].matured_dates, 60)
        self.assertTrue(a.loc[a.as_of_date.le(pd.Timestamp('2025-07-31')), 'alpha'].eq(.3).all())
        update = pd.Timestamp('2025-09-30')
        changed = source.copy()
        changed.loc[changed.label_end_date.gt(update), 'actual_signal'] *= -1000
        b, blog = build_dynamic_predictions(changed, **args)
        cols = ['alpha', 'final_signal']
        np.testing.assert_array_equal(a.loc[a.as_of_date.le(update), cols], b.loc[b.as_of_date.le(update), cols])
        pd.testing.assert_frame_equal(log[log.update_date.le(update)], blog[blog.update_date.le(update)])
        for row in log.itertuples():
            eligible = source[(source.as_of_date < row.update_date) & (source.label_end_date <= row.update_date)]
            last_dates = sorted(eligible.as_of_date.unique())[-60:]
            self.assertEqual(row.matured_rows, int(eligible.as_of_date.isin(last_dates).sum()))
            np.testing.assert_allclose(a.loc[a.as_of_date.eq(row.next_test_date), 'alpha'], row.alpha_after)

    def test_rank_gaussian_preserves_order_and_is_centered(self):
        transformed = rank_gaussian(np.array([30.0, 10.0, 20.0]))
        self.assertLess(transformed[1], transformed[2])
        self.assertLess(transformed[2], transformed[0])
        self.assertAlmostEqual(float(transformed.mean()), 0.0, places=12)

    def test_hedge_penalizes_the_model_with_higher_loss(self):
        result = update_hedge_weights(1.0, 1.0, static_loss=0.2, continual_loss=0.8, eta=0.5)
        self.assertLess(result["alpha"], 0.5)
        self.assertGreater(result["static_weight"], result["continual_weight"])

    def test_invalid_loss_keeps_weights_and_clips_alpha(self):
        result = update_hedge_weights(
            1.0, 1.0, static_loss=np.nan, continual_loss=0.2, eta=0.5,
            alpha_min=0.2, alpha_max=0.8,
        )
        self.assertEqual(result["static_weight"], 1.0)
        self.assertEqual(result["continual_weight"], 1.0)
        self.assertEqual(result["alpha"], 0.5)

    def test_month_end_update_applies_after_that_days_prediction(self):
        dates = pd.to_datetime([
            "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-04",
        ])
        rows = []
        for date_index, date in enumerate(dates):
            for stock_index, stock in enumerate(["A", "B", "C", "D"]):
                # Static ranks A>B>C>D while continual reverses them.  The
                # labels make the continual model preferable in the mature
                # July window, so alpha must change after July 31.
                rows.append({
                    "as_of_date": date,
                    "stock": stock,
                    "static_raw_signal": float(4 - stock_index),
                    "continual_raw_signal": float(stock_index + 1),
                    "actual_signal": float(stock_index + 1),
                    "label_end_date": date + pd.Timedelta(days=1),
                })
        joined = pd.DataFrame(rows)
        predictions, updates = build_dynamic_predictions(
            joined,
            test_start="2025-07-30",
            test_end="2025-08-04",
            history_days=60,
        )
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates.iloc[0]["update_date"], pd.Timestamp("2025-07-31"))
        before = predictions[predictions["as_of_date"] == pd.Timestamp("2025-07-31")]
        after = predictions[predictions["as_of_date"] == pd.Timestamp("2025-08-01")]
        self.assertTrue((before["alpha"] == 0.5).all())
        self.assertTrue((after["alpha"] > 0.5).all())
        self.assertEqual(updates.iloc[0]["next_test_date"], pd.Timestamp("2025-08-01"))

    def test_history_window_excludes_current_update_date(self):
        dates = pd.to_datetime(["2025-07-30", "2025-07-31", "2025-08-01"])
        rows = []
        for date in dates:
            for stock_index, stock in enumerate(["A", "B", "C"]):
                rows.append({
                    "as_of_date": date,
                    "stock": stock,
                    "static_raw_signal": float(stock_index),
                    "continual_raw_signal": float(2 - stock_index),
                    "actual_signal": float(stock_index),
                    "label_end_date": date,
                })
        _, updates = build_dynamic_predictions(
            pd.DataFrame(rows),
            test_start="2025-07-30",
            test_end="2025-08-01",
        )
        # July 31 may use July 30, but never July 31 itself.
        self.assertEqual(updates.iloc[0]["window_end_exclusive"], pd.Timestamp("2025-07-31"))
        self.assertLess(updates.iloc[0]["window_end_exclusive"], pd.Timestamp("2025-08-01"))
        self.assertEqual(updates.iloc[0]["matured_dates"], 1)

    def test_return_reference_uses_only_canonical_returns_and_validates_keys(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame({
                "as_of_date": ["2025-07-01"],
                "stock": ["AAA"],
                "pred_signal": [999.0],
                "actual_signal": [123.0],
                "actual_return_1": [0.02],
            }).to_csv(path, index=False)
            loaded = _load_return_reference(path)
        self.assertEqual(list(loaded.columns), ["as_of_date", "stock", "actual_return_1"])
        self.assertAlmostEqual(float(loaded.iloc[0]["actual_return_1"]), 0.02)

    def test_return_reference_rejects_duplicate_keys(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame({
                "as_of_date": ["2025-07-01", "2025-07-01"],
                "stock": ["AAA", "AAA"],
                "actual_return_1": [0.02, 0.03],
            }).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                _load_return_reference(path)

    def test_join_keeps_one_maturity_column_and_realized_signal(self):
        with TemporaryDirectory() as directory:
            static_path = Path(directory) / "static.csv"
            continual_path = Path(directory) / "continual.csv"
            common = {
                "as_of_date": ["2025-07-01"],
                "stock": ["AAA"],
                "actual_signal": [0.01],
                "label_end_date": ["2025-07-02"],
            }
            pd.DataFrame({**common, "pred_signal": [0.2]}).to_csv(static_path, index=False)
            pd.DataFrame({**common, "pred_signal": [0.3]}).to_csv(continual_path, index=False)
            joined = _load_and_join(static_path, continual_path)
        self.assertIn("actual_signal", joined.columns)
        self.assertIn("label_end_date", joined.columns)
        self.assertNotIn("label_end_date_x", joined.columns)
        self.assertNotIn("label_end_date_y", joined.columns)


if __name__ == "__main__":
    unittest.main()
