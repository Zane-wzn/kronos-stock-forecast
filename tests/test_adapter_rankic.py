import unittest

import numpy as np
import pandas as pd

from scripts.experiments.run_adapter_continual_csi300_90x10 import rankic
from scripts.experiments.run_adapter_continual_csi800_90x10 import rankic as rankic_csi800


class AdapterRankICTests(unittest.TestCase):
    def test_rankic_uses_per_sample_horizon_mean_and_original_row_alignment(self):
        meta = pd.DataFrame(
            {"as_of_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-01"] * 3)}
        )
        actual_signal = np.array([30, 10, 20, 2, 3, 1], dtype=float)
        actual = np.repeat((actual_signal / 10)[:, None], 10, axis=1)
        pred_signal = np.array([3, 1, 2, 2, 3, 1], dtype=float)
        pred = np.repeat(pred_signal[:, None], 10, axis=1)

        for metric in (rankic, rankic_csi800):
            with self.subTest(metric=metric.__module__):
                self.assertAlmostEqual(metric(pred, actual, meta), 1.0)

    def test_rankic_rejects_shape_or_index_misalignment(self):
        meta = pd.DataFrame({"as_of_date": pd.to_datetime(["2024-01-01"] * 2)})
        pred = np.ones((2, 10))
        actual = np.ones((2, 10))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            rankic(pred, actual[:, :9], meta)
        with self.assertRaisesRegex(ValueError, "RangeIndex"):
            rankic(pred, actual, meta.set_axis([1, 2]))


if __name__ == "__main__":
    unittest.main()
