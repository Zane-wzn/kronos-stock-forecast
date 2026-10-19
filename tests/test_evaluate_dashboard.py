from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import EvaluationConfig, evaluate_frame
from evaluate.run_evaluation import _write_result


ROOT = Path(__file__).resolve().parents[1]


class DashboardEvaluationTests(unittest.TestCase):
    def config(self, **kwargs):
        return EvaluationConfig(test_start=None, test_end=None, bootstrap_reps=0, **kwargs)

    def frame(self, days=7):
        return pd.DataFrame([
            {"as_of_date": f"2025-07-{day + 1:02d}", "stock": stock,
             "pred_signal": (4 - idx if day == 0 else idx), "actual_signal": (idx - 2) / 10,
             "actual_return_1": (idx - 2) / 100}
            for day in range(days) for idx, stock in enumerate("ABCD")
        ])

    def test_direction_and_zero_baseline(self):
        frame = self.frame(1)
        frame["pred_signal"] = [0.1, -0.1, 0.3, 0]
        frame["actual_signal"] = [0.2, -0.2, -0.1, 0]
        result = evaluate_frame(frame, self.config())["metrics"]["path"]
        self.assertAlmostEqual(result["point"]["mae"], 0.15)
        self.assertAlmostEqual(result["point"]["mase_vs_zero_return"], 1.2)
        direction = result["direction"]
        self.assertEqual((direction["tp"], direction["tn"], direction["fp"], direction["fn"]), (1, 2, 1, 0))
        self.assertAlmostEqual(direction["accuracy"], 0.75)
        self.assertAlmostEqual(direction["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(direction["f1"], 2 / 3)

    def test_no_endpoint_or_daily_return_is_not_zero(self):
        frame = self.frame().drop(columns="actual_return_1")
        result = evaluate_frame(frame, self.config())["metrics"]
        self.assertEqual(result["endpoint"]["point"]["status"], "无法验证")
        self.assertEqual(result["path"]["portfolio"]["status"], "无法验证")
        self.assertEqual(result["trajectory_pooled"]["status"], "无法验证")
        self.assertEqual(result["kronos_aer"]["status"], "无法验证")

    def test_holding_lock_drop_and_daily_not_multiday_returns(self):
        result = evaluate_frame(self.frame(), self.config(top_k=1, drop_n=1, min_holding_days=5))
        daily = result["daily"]["path"]["portfolio"]
        self.assertTrue(all(row["top_raw"] == -0.02 for row in daily[:5]))
        self.assertAlmostEqual(daily[5]["top_raw"], 0.01)
        self.assertEqual([row["sold"] for row in daily], [0, 0, 0, 0, 0, 1, 0])
        self.assertEqual([row["turnover"] for row in daily], [1, 0, 0, 0, 0, 1, 0])

    def test_drawdown_includes_initial_capital(self):
        frame = self.frame(1)
        frame["actual_return_1"] = -0.1
        result = evaluate_frame(frame, self.config(cost_rate=0))["metrics"]["path"]["portfolio"]
        self.assertAlmostEqual(result["max_drawdown"], -0.1)

    def test_nonfinite_pair_filter_before_ranking(self):
        frame = self.frame(1)
        frame["pred_signal"] = [1, 2, 3, 100]
        frame["actual_signal"] = [1, 2, 3, np.inf]
        result = evaluate_frame(frame, self.config())["metrics"]["path"]
        self.assertEqual(result["point"]["n"], 3)
        self.assertAlmostEqual(result["cross_section"]["rankic"]["mean"], 1)

    def test_prices_convert_and_partial_trajectory_is_unavailable(self):
        frame = self.frame(1).drop(columns=["pred_signal", "actual_signal", "actual_return_1"])
        frame["current_close"] = 100
        for horizon in range(1, 11):
            frame[f"pred_close_{horizon}"] = 100 + horizon
            frame[f"actual_close_{horizon}"] = 100 + 2 * horizon
        result = evaluate_frame(frame, self.config())["metrics"]
        self.assertAlmostEqual(result["trajectory_pooled"]["mae"], 0.055)
        self.assertAlmostEqual(result["endpoint"]["point"]["mae"], 0.1)
        self.assertAlmostEqual(result["path"]["point"]["mae"], 0.055)
        frame = frame.drop(columns="pred_close_4")
        result = evaluate_frame(frame, self.config())["metrics"]
        self.assertEqual(result["trajectory_pooled"]["status"], "无法验证")

    def test_duplicate_keys_and_empty_window_fail(self):
        frame = self.frame(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_frame(pd.concat([frame, frame]), self.config())
        with self.assertRaisesRegex(ValueError, "test window"):
            evaluate_frame(frame, EvaluationConfig(test_start="2030-01-01", test_end="2030-12-31"))
        with self.assertRaisesRegex(ValueError, "test_start"):
            evaluate_frame(frame, EvaluationConfig(test_start="2030-01-01"))

    def test_bootstrap_is_reproducible_and_records_custom_block(self):
        frame = self.frame()
        config = EvaluationConfig(test_start=None, test_end=None, bootstrap_reps=100, bootstrap_block=3, seed=7)
        first = evaluate_frame(frame, config)["metrics"]["path"]["cross_section"]["rankic"]
        second = evaluate_frame(frame, config)["metrics"]["path"]["cross_section"]["rankic"]
        self.assertEqual(first["moving_block_bootstrap_ci95"], second["moving_block_bootstrap_ci95"])
        self.assertEqual(first["bootstrap_block"], 3)
        low, high = first["moving_block_bootstrap_ci95"]
        self.assertLessEqual(low, high)

    def test_forced_exit_while_locked(self):
        frame = self.frame(2)
        frame = frame[~((frame.as_of_date == "2025-07-02") & (frame.stock == "A"))]
        result = evaluate_frame(frame, self.config(top_k=1, drop_n=0, min_holding_days=5))
        daily = result["daily"]["path"]["portfolio"]
        self.assertEqual(daily[1]["forced_exits"], 1)
        self.assertEqual(daily[1]["sold"], 1)

    def test_aer_external_report_and_gate_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            source, report, log = home / "pred.csv", home / "aer.csv", home / "updates.csv"
            frame = self.frame(2)
            frame.to_csv(source, index=False)
            pd.DataFrame({"as_of_date": ["2025-07-01", "2025-07-02"], "return": [0.01, 0.02],
                          "benchmark": [0.002, 0.003], "cost": [0.001, 0.001]}).to_csv(report, index=False)
            pd.DataFrame({"accepted": [True], "validation_rankic": [0.1]}).to_csv(log, index=False)
            output = home / "out"
            output.mkdir()
            info = _write_result(frame, [source], output, self.config(), aer_report=report, update_log=log)
            summary = json.loads(Path(info["summary"]).read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["metrics"]["kronos_aer"]["aer"], 238 * 0.0115)
            self.assertFalse(summary["metrics"]["validation_rankic_gate"]["is_test_metric"])
            self.assertEqual(len(summary["sources"][0]["sha256"]), 64)

    def test_cli_batch_no_stem_collision_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            files = []
            for label in ("one", "two"):
                source = home / label / "predictions.csv"
                source.parent.mkdir()
                self.frame(2).to_csv(source, index=False)
                files.append(str(source))
            output = home / "out"
            command = [sys.executable, str(ROOT / "evaluate/run_evaluation.py"), "--predictions", *files,
                       "--output-dir", str(output), "--bootstrap-reps", "0"]
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            index = json.loads((output / "evaluation_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index["results"]), 2)
            self.assertNotEqual(index["results"][0]["summary"], index["results"][1]["summary"])
            repeated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
            self.assertNotEqual(repeated.returncode, 0)

    def test_cli_fixed_panel_aligns_and_rejects_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            source, reference = home / "predictions.csv", home / "keys.csv"
            self.frame(2).to_csv(source, index=False)
            keys = self.frame(1)[["as_of_date", "stock"]]
            keys.to_csv(reference, index=False)
            output = home / "aligned"
            command = [sys.executable, str(ROOT / "evaluate/run_evaluation.py"), "--predictions", str(source),
                       "--reference-keys", str(reference), "--output-dir", str(output), "--bootstrap-reps", "0"]
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["alignment"]["input_rows"], 8)
            self.assertEqual(summary["alignment"]["aligned_rows"], 4)
            keys.loc[len(keys)] = ["2025-07-01", "MISSING"]
            keys.to_csv(reference, index=False)
            command[command.index(str(output))] = str(home / "missing")
            failed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("1", failed.stderr)


if __name__ == "__main__":
    unittest.main()
