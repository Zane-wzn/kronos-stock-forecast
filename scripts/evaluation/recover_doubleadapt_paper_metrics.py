"""Re-evaluate recovered DoubleAdapt forecasts; preserve missing-run-evidence limits."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from evaluate.inputs import normalize_predictions
from evaluate.metrics import EvaluationConfig, clean_json, evaluate_frame
from src.doubleadapt.compare import return_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefer-archives", action="store_true", help="Recompute from Git LFS ZIPs even when original local files exist")
    args = parser.parse_args()
    out = ROOT / "docs/paper_external_baselines_20261106"
    out.mkdir(exist_ok=True)
    root = ROOT / "experiments/csi300_aligned/models"
    keys = ["as_of_date", "stock"]
    pc = [f"pred_return_{h}" for h in range(1, 11)]
    ac = [f"actual_return_{h}" for h in range(1, 11)]
    ref_path = root / "prediction_adapter/predictions.csv"
    if ref_path.exists() and not args.prefer_archives:
        reference = pd.read_csv(ref_path, dtype={"stock": str})
    else:
        with zipfile.ZipFile(ROOT / "docs/baseline_comparison_20261105/prediction_archives/Prediction_static.zip") as archive:
            with archive.open("predictions.csv") as stream:
                reference = pd.read_csv(stream, dtype={"stock": str})
    ref = normalize_predictions(reference)
    ref = ref.sort_values(keys).reset_index(drop=True)
    cfg = EvaluationConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=.0015,
                           annualization_days=252, bootstrap_reps=0)
    rows, audits = [], []
    for seed in (100, 101, 102):
        path = root / f"doubleadapt_gru_90x10_suite_20261013/seed{seed}/predictions.csv"
        if path.exists() and not args.prefer_archives:
            data = pd.read_csv(path, dtype={"stock": str}, float_precision="round_trip")
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            with zipfile.ZipFile(out / "archives" / f"DoubleAdapt_seed{seed}.zip") as archive:
                original = archive.read("predictions.csv")
                source_hash = hashlib.sha256(original).hexdigest()
                import io
                data = pd.read_csv(io.BytesIO(original), dtype={"stock": str}, float_precision="round_trip")
        f = normalize_predictions(data)
        f = f.sort_values(keys).reset_index(drop=True)
        assert len(f) == 66103 and f.as_of_date.nunique() == 226
        assert not f.duplicated(keys).any() and f[keys].equals(ref[keys])
        assert np.isfinite(f[pc + ac].to_numpy()).all()
        error = float(np.abs(f[ac].to_numpy() - ref[ac].to_numpy()).max())
        assert error < 1e-7
        f[ac] = ref[ac].to_numpy()
        for prefix, cols in (("pred", pc), ("actual", ac)):
            f[prefix + "_signal"] = f[cols].mean(axis=1)
            f[prefix + "_endpoint_return"] = f[cols[-1]]
        result = evaluate_frame(f, cfg)
        (out / f"DoubleAdapt_seed{seed}_evaluation.json").write_text(
            json.dumps(clean_json(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for mode in ("path", "endpoint"):
            m = result["metrics"][mode]
            d = pd.DataFrame(result["daily"][mode]["portfolio"])
            d.to_csv(out / f"DoubleAdapt_seed{seed}_{mode}_daily.csv", index=False)
            rows.append({"model": "DoubleAdapt_GRU", "seed": seed, "signal": mode,
                         "rows": len(f), "dates": f.as_of_date.nunique(),
                         "trajectory_rmse": result["metrics"]["trajectory_pooled"]["rmse"],
                         "trajectory_mae": result["metrics"]["trajectory_pooled"]["mae"],
                         "ic": m["cross_section"]["ic"]["mean"],
                         "rankic": m["cross_section"]["rankic"]["mean"],
                         **return_stats(d.top_net.to_numpy())})
        evidence = {name: (path.parent / name).exists() for name in ("run_config.json", "status.json")}
        audits.append({"seed": seed, "source": str(path.relative_to(ROOT)).replace(chr(92), "/"),
                       "sha256": source_hash,
                       "rows": len(f), "dates": f.as_of_date.nunique(), "duplicate_keys": 0,
                       "matches_reference_keys": True, "target_max_abs_error": error,
                       "all_prediction_and_target_values_finite": True,
                       "run_evidence_present": evidence,
                       "qualification": "Recovered full forecasts; training provenance/configuration not fully verifiable. Not certified official reproduction."})
        print(f"Recovered seed{seed}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "doubleadapt_metrics.csv", index=False)
    numeric = ["trajectory_rmse", "trajectory_mae", "ic", "rankic", "net_cumulative_return", "sharpe", "max_drawdown"]
    summary = metrics.groupby("signal")[numeric].agg(["mean", "std"])
    summary.to_csv(out / "doubleadapt_seed_summary.csv")
    (out / "doubleadapt_recovery_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
