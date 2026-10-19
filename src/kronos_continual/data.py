"""Strict data identity, causal raw-window summaries, immutable file helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = "111f72975e77e165211b441a731b4da17dca4e26629b848839ce13aaffcd94eb"
KEYS = ["as_of_date", "stock"]
PCOLS = [f"pred_return_{k}" for k in range(1, 11)]
ACOLS = [f"actual_return_{k}" for k in range(1, 11)]
MCOLS = ["split", "stock", "as_of_date", "input_start_date", "input_end_date", "label_start_date", "label_end_date"]
FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    obj.to_csv(tmp, index=False)
    os.replace(tmp, path)


def npy_info(path, shape=None):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path.resolve()), "valid": False, "reason": "missing"}
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        actual_shape, _, dtype = reader(f)
        expected_bytes = f.tell() + int(np.prod(actual_shape))*dtype.itemsize
    size = path.stat().st_size
    valid = size == expected_bytes and (shape is None or tuple(shape) == tuple(actual_shape))
    return {"path": str(path.resolve()), "valid": valid, "shape": list(actual_shape), "dtype": str(dtype),
            "bytes": size, "expected_bytes": expected_bytes, "sha256": sha256(path),
            "reason": None if valid else "truncated_or_wrong_shape"}


def load_meta(cache, panel):
    meta = pd.read_csv(Path(cache)/"meta.csv", dtype=str)
    locked = pd.read_csv(Path(panel)/"eligible_stock_date_pairs.csv", usecols=MCOLS, dtype=str)
    if not meta[MCOLS].equals(locked[MCOLS]):
        raise AssertionError("metadata keys/order/boundaries differ from locked panel")
    if meta.duplicated(KEYS).any():
        raise AssertionError("duplicate panel keys")
    for split in ["train", "val", "test"]:
        part = meta[meta.split.eq(split)]
        if not part[KEYS].equals(part.sort_values(KEYS)[KEYS]):
            raise AssertionError(f"{split} rows must follow cache builder's date/stock order")
    test = meta[meta.split.eq("test")]
    if len(test) != 66103 or test.as_of_date.nunique() != 226 or test.as_of_date.min() != "2025-07-01" or test.as_of_date.max() != "2026-06-05":
        raise AssertionError("test must cover exactly 226 dates and 66,103 keys")
    if (meta.input_end_date != meta.as_of_date).any() or (meta.label_start_date <= meta.as_of_date).any() or (meta.label_end_date < meta.label_start_date).any():
        raise AssertionError("invalid causal input/label interval")
    if meta.loc[meta.split.eq("train"), "as_of_date"].max() > "2024-12-15":
        raise AssertionError("training input date exceeds cutoff")
    val = meta[meta.split.eq("val")]
    if val.as_of_date.min() < "2025-01-02" or val.as_of_date.max() > "2025-06-14":
        raise AssertionError("validation date exceeds locked boundaries")
    meta["row_id"] = np.arange(len(meta))
    return meta


def base_predict(model, path, device, rows=None):
    import torch
    h = np.load(path, mmap_mode="r")
    ids = np.arange(len(h)) if rows is None else np.asarray(rows)
    out = np.empty((len(ids), 10), np.float32)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(ids), 256):
            x = np.array(h[ids[b:b+256]], dtype=np.float32, copy=True)
            if not np.isfinite(x).all():
                raise FloatingPointError("nonfinite hidden state")
            out[b:b+256] = model(torch.tensor(x, device=device)).cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite frozen prediction")
    return out


def preflight(args):
    """Runs even with missing train/val files; never performs any adaptation."""
    import torch
    from .page1c_mars_cl import Page1CModel, ResidualAdapter
    meta = load_meta(args.cache_dir, args.panel_dir)
    checkpoint_hash = sha256(args.initial_checkpoint)
    expected_hash = (args.expected_checkpoint_sha256 or EXPECTED_HASH).lower()
    if checkpoint_hash.lower() != expected_hash:
        raise AssertionError("Page1-C checkpoint hash does not match the locked expected hash")
    arrays = {}
    for split in ["train", "val", "test"]:
        n = int(meta.split.eq(split).sum())
        arrays[f"hidden_{split}.npy"] = npy_info(args.hidden_dir/f"hidden_{split}.npy", (n, 512))
        current = args.hidden_dir/f"current_{split}.npy"
        arrays[f"current_{split}.npy"] = npy_info(current, (n,))
    arrays["y.npy"] = npy_info(args.cache_dir/"y.npy", (len(meta), 10))
    blockers = [f"{k}: {v['reason']}" for k, v in arrays.items() if not v["valid"]]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; device is never silently changed")
    model = Page1CModel().to(device)
    model.load_state_dict(torch.load(args.initial_checkpoint, weights_only=True, map_location=device), strict=True)
    model.eval().requires_grad_(False)
    test = meta[meta.split.eq("test")]
    historical_path = args.historical_predictions or (args.initial_checkpoint.parent/"predictions.csv")
    historical_path = Path(historical_path)
    historical = pd.read_csv(historical_path, dtype={"stock": str})
    if not historical[KEYS].equals(test[KEYS].reset_index(drop=True)) or historical.duplicated(KEYS).any():
        raise AssertionError("historical Page1-C test keys/order differ")
    identity = {"checkpoint_sha256": checkpoint_hash, "expected_checkpoint_sha256": expected_hash,
                "historical_predictions_path": str(historical_path.resolve()),
                "historical_predictions_sha256": sha256(historical_path),
                "test_keys_equal": True, "rows": len(test), "dates": test.as_of_date.nunique(), "max_prediction_error": None}
    if arrays["hidden_test.npy"]["valid"]:
        base = base_predict(model, args.hidden_dir/"hidden_test.npy", device)
        error = float(np.abs(base-historical[PCOLS].to_numpy()).max())
        identity["max_prediction_error"] = error
        if error > 1e-6:
            blockers.append(f"Page1-C reproduction max error {error} exceeds 1e-6")
        with torch.no_grad():
            zero = ResidualAdapter().to(device).eval()
            b = torch.tensor(base, device=device)
            z = zero(b, torch.zeros((len(b), 18), device=device)).cpu().numpy()
        identity["zero_residual_max_error"] = float(np.abs(z-base).max())
        if identity["zero_residual_max_error"] > 1e-7:
            blockers.append("zero initialization identity failed")
    if arrays["y.npy"]["valid"]:
        y = np.load(args.cache_dir/"y.npy", mmap_mode="r")
        err = float(np.abs(np.asarray(y[test.row_id])-historical[ACOLS].to_numpy()).max())
        identity["label_max_error"] = err
        if err > 1e-7:
            blockers.append("historical test labels conflict with locked label cache")
    for target, path in [("pred_signal", historical[PCOLS].mean(axis=1)), ("pred_endpoint_return", historical[PCOLS[-1]])]:
        # The historical 28-column file predates the explicit endpoint column.
        # Its endpoint is unambiguously the persisted tenth prediction.
        if target in historical and not np.allclose(historical[target], path, rtol=0, atol=1e-7):
            blockers.append(f"historical {target} semantics mismatch")
    # Current close arrays independently anchor split row order to raw stock data.
    raw_manifest = {}
    split_meta = {s: meta[meta.split.eq(s)].reset_index(drop=True) for s in ["train", "val", "test"]}
    current_arrays = {s: np.load(args.hidden_dir/f"current_{s}.npy", mmap_mode="r") for s in split_meta if arrays[f"current_{s}.npy"]["valid"]}
    for stock in meta.stock.unique():
        path = args.data_root/f"{stock.replace('.', '_')}.csv"
        if not path.exists():
            blockers.append(f"raw stock file absent: {stock}")
            continue
        f = pd.read_csv(path, dtype={"date": str})
        if f.date.duplicated().any() or not f.date.is_monotonic_increasing:
            raise AssertionError(f"invalid raw date order: {stock}")
        raw_manifest[stock] = {"path": str(path.resolve()), "sha256": sha256(path), "amount_source": "raw" if "amount" in f else "OHLC_mean_times_volume_as_in_hidden_cache_builder"}
        close = f.set_index("date").close
        for split in ["train", "val", "test"]:
            if not arrays[f"current_{split}.npy"]["valid"]:
                continue
            sm = split_meta[split]
            part = sm[sm.stock.eq(stock)]
            current = current_arrays[split]
            observed = np.asarray(current[part.index])
            expected = close.reindex(part.as_of_date).to_numpy(np.float32)
            if not np.array_equal(observed, expected):
                raise AssertionError(f"{split} hidden current-close row alignment fails: {stock}")
    result = {"status": "ready" if not blockers else "blocked", "blockers": blockers, "arrays": arrays,
              "identity": identity, "raw_manifest": raw_manifest, "panel_rows": len(meta),
              "torch": torch.__version__, "device": str(device), "checkpoint_sha256": checkpoint_hash,
              "input_hashes": {str(p.resolve()): sha256(p) for p in [args.cache_dir/"meta.csv", args.panel_dir/"eligible_stock_date_pairs.csv", args.panel_dir/"date_windows.csv"]}}
    return result, meta, model


def history_pool(meta):
    """Fixed seed independent of run seed. Only labels mature by train cutoff."""
    eligible = meta[meta.split.eq("train") & meta.label_end_date.le("2024-12-15")]
    cap = min(60000, len(eligible))
    rng = np.random.default_rng(20261013)
    years = eligible.as_of_date.str[:4]
    counts = years.value_counts().sort_index()
    exact = counts/counts.sum()*cap
    quotas = np.floor(exact).astype(int)
    for year in (exact-quotas).sort_values(ascending=False, kind="stable").index[:cap-quotas.sum()]:
        quotas[year] += 1
    ids = np.concatenate([rng.choice(eligible.index[years.eq(y)], int(quotas[y]), replace=False) for y in quotas.index])
    return np.sort(ids)


def mature_partition(test, date):
    mature = test[test.as_of_date.lt(date) & test.label_end_date.le(date)]
    expected = test.groupby("as_of_date").size()
    observed = mature.groupby("as_of_date").size()
    complete = sorted(observed.index[observed.eq(expected.reindex(observed.index))])
    shadow_dates = complete[-10:]
    update_dates = complete[:-10][-30:]
    return mature[mature.as_of_date.isin(update_dates)], mature[mature.as_of_date.isin(shadow_dates)]


def raw_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (90, 6) or not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError("invalid raw 90x6 window")
    return np.concatenate([values.mean(0), values.std(0), values[-1]])


def build_states(args, meta, out):
    """Raw OHLCV + cache-builder amount, then train-only standardization; no clipping."""
    out = Path(out)
    if (out/"state_manifest.json").exists():
        manifest = json.loads((out/"state_manifest.json").read_text(encoding="utf-8"))
        if sha256(out/"state.npy") != manifest["state_sha256"]:
            raise AssertionError("state cache corrupted")
        return np.load(out/"state.npy", mmap_mode="r"), manifest
    out.mkdir(parents=True, exist_ok=True)
    state = np.empty((len(meta), 18), np.float64)
    # The locked panel has stock-specific calendars; never collapse by date.
    frames = {}
    for stock in meta.stock.unique():
        f = pd.read_csv(args.data_root/f"{stock.replace('.', '_')}.csv", dtype={"date": str})
        if "amount" not in f:
            f["amount"] = f[["open", "high", "low", "close"]].mean(axis=1)*f.volume
        else:
            f["amount"] = f.amount.fillna(f[["open", "high", "low", "close"]].mean(axis=1)*f.volume)
        frames[stock] = (dict(zip(f.date, range(len(f)))), f[FIELDS].to_numpy(np.float64))
    seen = 0
    # Lean metadata and explicit input dates only: no future closes are read here.
    cols = ["split", "stock", "as_of_date", "input_dates", "input_calendar_hash"]
    for chunk in pd.read_csv(args.panel_dir/"eligible_stock_date_pairs.csv", usecols=cols, dtype=str, chunksize=10000):
        for row in chunk.itertuples():
            i = int(row.Index)
            if tuple(meta.loc[i, ["split", "stock", "as_of_date"]]) != (row.split, row.stock, row.as_of_date):
                raise AssertionError("state reconstruction row mismatch")
            dates = row.input_dates.split("|")
            if len(dates) != 90 or dates[-1] != row.as_of_date or any(a >= b for a,b in zip(dates, dates[1:])) or hashlib.sha256(row.input_dates.encode()).hexdigest() != row.input_calendar_hash:
                raise AssertionError("state input calendar mismatch")
            positions, values = frames[row.stock]
            state[i] = raw_summary(values[[positions[d] for d in dates]])
        seen += len(chunk)
        if seen % 100000 == 0:
            print(json.dumps({"stage": "raw_state", "rows": seen}), flush=True)
    if seen != len(meta):
        raise AssertionError("incomplete state inputs")
    training = state[meta.split.eq("train")]
    center, scale = training.mean(0), training.std(0)
    scale[scale < 1e-12] = 1.
    state = ((state-center)/scale).astype(np.float32)
    if not np.isfinite(state).all():
        raise FloatingPointError("nonfinite normalized states")
    np.save(out/"state.npy", state)
    manifest = {"center": center.tolist(), "scale": scale.tolist(), "fit_split": "train", "fit_rows": len(training),
                "fields": FIELDS, "statistics_order": ["mean", "std_ddof0", "last"], "clip": None,
                "amount_rule": "raw or OHLC_mean*volume; identical to hidden cache builder; raw files in data_manifest",
                "state_sha256": sha256(out/"state.npy")}
    atomic_json(out/"state_manifest.json", manifest)
    return state, manifest
