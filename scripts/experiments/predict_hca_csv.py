"""Run the archived HCA model on one 90-day OHLCV CSV."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from scripts.data.build_continual_csi300_hidden_cache import load_predictor, hidden_state, normalise_ohlcv, stamps
from src.kronos_continual.hca_portable import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=ROOT / "models/hca_best/hca_pca_final.safetensors")
    p.add_argument("--kronos-repo", type=Path, default=ROOT / "third_party/Kronos_official_src/Kronos-master")
    p.add_argument("--model-name", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer-name", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.csv)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for c in ("open", "high", "low", "close", "volume"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if "amount" not in frame:
        frame["amount"] = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
    else:
        fallback = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(fallback)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume", "amount"])
    if len(frame) < 90:
        raise ValueError(f"Need at least 90 valid rows, got {len(frame)}")
    context = frame.tail(90).copy()
    if not np.isfinite(context[["open", "high", "low", "close", "volume", "amount"]].to_numpy()).all():
        raise ValueError("Nonfinite values in the input window")
    from huggingface_hub import snapshot_download
    if args.model_name == "NeoQuasar/Kronos-small":
        args.model_name = snapshot_download(args.model_name, revision="901c26c1332695a2a8f243eb2f37243a37bea320")
    if args.tokenizer_name == "NeoQuasar/Kronos-Tokenizer-base":
        args.tokenizer_name = snapshot_download(args.tokenizer_name, revision="0e0117387f39004a9016484a186a908917e22426")
    ns = SimpleNamespace(kronos_repo=str(args.kronos_repo), model_name=args.model_name,
                         tokenizer_name=args.tokenizer_name, device=args.device)
    predictor = load_predictor(ns)
    x = normalise_ohlcv(context)
    ts = stamps(context["date"])
    with torch.inference_mode():
        hidden = hidden_state(predictor, np.expand_dims(x, 0), np.expand_dims(ts, 0)).cpu()
    model = load_model(args.weights, device=args.device)
    with torch.inference_mode():
        values = model(hidden.to(args.device))[0].cpu().numpy()
    result = {"as_of_date": str(context.date.iloc[-1].date()),
              "input_start": str(context.date.iloc[0].date()),
              "input_end": str(context.date.iloc[-1].date()),
              "weights": str(args.weights),
              "predicted_cumulative_return": {str(k): float(v) for k, v in enumerate(values, 1)},
              "path_signal": float(values.mean()), "endpoint_signal": float(values[-1])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
