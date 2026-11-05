"""Predict ten cumulative returns from [N,512] frozen Kronos hidden features."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.kronos_continual.hca_portable import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/hca_best/hca_pca_final.safetensors")
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hidden = np.load(args.hidden, mmap_mode="r", allow_pickle=False)
    if hidden.ndim != 2 or hidden.shape[1] != 512:
        raise ValueError("Expected hidden features with shape [N,512]")
    model = load_model(args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        handle.write(",".join([f"pred_return_{k}" for k in range(1, 11)] + ["path_signal", "endpoint_signal"]) + "\n")
        with torch.inference_mode():
            for start in range(0, len(hidden), 1024):
                batch = np.array(hidden[start:start + 1024], dtype=np.float32)
                if not np.isfinite(batch).all():
                    raise ValueError("Nonfinite hidden features")
                pred = model(torch.from_numpy(batch)).numpy()
                np.savetxt(handle, np.column_stack((pred, pred.mean(1), pred[:, -1])), delimiter=",", fmt="%.10g")
    print(f"Saved {len(hidden)} predictions to {args.output}")


if __name__ == "__main__":
    main()
