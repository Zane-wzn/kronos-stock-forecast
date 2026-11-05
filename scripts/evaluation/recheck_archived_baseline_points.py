"""Verify portable forecast archives and recompute metrics without raw prices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from evaluate.metrics import evaluate_frame, EvaluationConfig


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", type=Path, default=ROOT / "docs/baseline_comparison_20261105")
    p.add_argument("--model", default="Chronos2")
    p.add_argument("--verify-all-zips", action="store_true")
    args = p.parse_args()
    out = args.archive_dir
    if args.verify_all_zips:
        manifest = json.loads((out / "archive_manifest.json").read_text(encoding="utf-8"))
        for a in manifest["archives"]:
            path = out / a["file"]
            assert sha256(path) == a["sha256"], path
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, path
                for member in a["members"]:
                    h = hashlib.sha256()
                    with archive.open(member["name"]) as stream:
                        for b in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(b)
                    assert h.hexdigest() == member["sha256"], member["name"]
        print("All seven original prediction archives verified")
    names = {"Kronos_zero_shot", "Chronos2", "TimesFM3", "Prediction_static", "Prediction_online", "HCA_PCA"}
    if args.model not in names:
        raise ValueError(f"Choose one of {sorted(names)}")
    frame = pd.read_csv(out / "normalized_points" / f"{args.model}.csv.gz", dtype={"stock": str}, float_precision="round_trip")
    assert len(frame) == 66103 and frame.as_of_date.nunique() == 226
    cfg = EvaluationConfig(top_k=30, drop_n=3, min_holding_days=5, cost_rate=.0015, annualization_days=252, bootstrap_reps=0)
    result = evaluate_frame(frame, cfg)
    expected = pd.read_csv(out / "full_226_metrics.csv")
    checks = []
    for signal in ("path", "endpoint"):
        r = expected.loc[expected.model.eq(args.model) & expected.signal.eq(signal)].iloc[0]
        m = result["metrics"][signal]
        net = np.asarray([x["top_net"] for x in result["daily"][signal]["portfolio"]])
        values = {"net_cumulative_return": float(np.prod(1 + net) - 1),
                  "ic": m["cross_section"]["ic"]["mean"], "rankic": m["cross_section"]["rankic"]["mean"],
                  "trajectory_rmse": result["metrics"]["trajectory_pooled"]["rmse"]}
        for key, value in values.items():
            assert abs(value - r[key]) < 1e-12, (key, value, r[key])
        checks.append({"signal": signal, **values})
    print(json.dumps({"model": args.model, "status": "passed", "checks": checks}))


if __name__ == "__main__":
    main()
