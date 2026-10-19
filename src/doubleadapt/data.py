"""Locked stock windows and exact train-only robust normalization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["as_of_date", "stock"]
FIELDS = ["open", "high", "low", "close", "volume"]
TRAIN_END, TEST_START = "2024-12-15", "2025-07-01"
TEST_END = "2026-06-05"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def validate_test(meta, expected_rows=66103, expected_dates=226):
    test = meta[meta.split.eq("test")]
    if len(test) != expected_rows or test.as_of_date.nunique() != expected_dates:
        raise AssertionError("Locked test panel must contain 66,103 rows / 226 dates")
    if test.as_of_date.min() != TEST_START or test.as_of_date.max() != TEST_END:
        raise AssertionError("Unexpected test boundaries")
    if meta.duplicated(KEYS).any():
        raise AssertionError("Duplicate stock/date keys")


def raw_window(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (90, 5) or not np.isfinite(values).all() or (values[:, :4] <= 0).any():
        raise ValueError("Invalid locked OHLCV window")
    if (values[:, 4] < 0).any():
        raise ValueError("Negative volume")
    normalized = values.copy()
    normalized[:, :4] /= values[-1, 3]
    mean = values[:, 4].mean()
    normalized[:, 4] = values[:, 4] / mean if mean > 0 else 0
    return normalized.astype(np.float32), mean == 0


def fit_robust(x, train_ids):
    centers, scales = [], []
    for feature in range(5):
        # Per-channel temporary array: exact median over all train windows.
        values = np.asarray(x[train_ids, :, feature]).reshape(-1).copy()
        center = float(np.median(values, overwrite_input=True))
        values = np.asarray(x[train_ids, :, feature]).reshape(-1).copy()
        np.subtract(values, center, out=values)
        np.abs(values, out=values)
        scale = float(np.median(values, overwrite_input=True)) * 1.4826
        centers.append(center)
        scales.append(scale if scale > 1e-12 else 1.)
    return np.array(centers, dtype=np.float32), np.array(scales, dtype=np.float32)


def normalize(x, center, scale, zero_volume=None):
    result = np.clip((x-center)/scale, -3, 3).astype(np.float32)
    if zero_volume is not None:
        result[np.asarray(zero_volume, dtype=bool), :, 4] = 0
    return result


def build_cache(data_root, reference_dir, cache_dir):
    cache_dir = Path(cache_dir)
    pair_path = Path(reference_dir)/"eligible_stock_date_pairs.csv"
    manifest = {"protocol_version":2,"pairs_sha256": sha256(pair_path), "raw_files": {}}
    stocks = pd.read_csv(pair_path, usecols=["stock"]).stock.unique()
    for stock in stocks:
        source = Path(data_root)/f"{stock.replace('.', '_')}.csv"
        manifest["raw_files"][stock] = {"path": str(source.resolve()), "sha256": sha256(source)}
    if (cache_dir/"complete.json").exists():
        prior = json.loads((cache_dir/"manifest.json").read_text(encoding="utf-8"))
        if prior != manifest:
            raise AssertionError("Data changed since cache was created")
        return load_cache(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"Incomplete cache; use a new directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Only lean metadata is kept in RAM; the 810 MB date lists are read in chunks.
    columns = ["split", "stock", "as_of_date", "input_start_date", "input_end_date",
               "label_start_date", "label_end_date"]
    meta = pd.read_csv(pair_path, usecols=columns, dtype=str)
    validate_test(meta)
    meta["row_id"] = np.arange(len(meta))
    meta.to_csv(cache_dir/"meta.csv", index=False)
    n = len(meta)
    x = np.lib.format.open_memmap(cache_dir/"x.npy", mode="w+", dtype="float32", shape=(n,90,5))
    y = np.lib.format.open_memmap(cache_dir/"y.npy", mode="w+", dtype="float64", shape=(n,10))
    zeros = np.zeros(n, dtype=bool)
    frames = {}
    for stock, entry in manifest["raw_files"].items():
        frame = pd.read_csv(entry["path"], dtype={"date": str, "code": str})
        if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise ValueError(f"Invalid date order/duplicates: {stock}")
        if set(frame.code) != {stock}:
            raise ValueError(f"Stock identity mismatch: {stock}")
        frames[stock] = (dict(zip(frame.date, range(len(frame)))), frame[FIELDS].to_numpy(dtype=float))
    offset = 0
    for chunk in pd.read_csv(pair_path, dtype=str, chunksize=15000):
        for stock, group in chunk.groupby("stock", sort=False):
            positions, values = frames[stock]
            for row in group.itertuples():
                i = int(row.Index)
                dates, labels = row.input_dates.split("|"), row.label_dates.split("|")
                if len(dates) != 90 or len(labels) != 10 or dates[-1] != row.as_of_date:
                    raise AssertionError("Locked date list mismatch")
                if dates[0] != row.input_start_date or dates[-1] != row.input_end_date or (
                    labels[0] != row.label_start_date or labels[-1] != row.label_end_date):
                    raise AssertionError("Locked window boundary mismatch")
                if hashlib.sha256(row.input_dates.encode()).hexdigest()!=row.input_calendar_hash or (
                    hashlib.sha256(row.label_dates.encode()).hexdigest()!=row.label_calendar_hash):
                    raise AssertionError("Locked date-list hash mismatch")
                if not all(a < b for a, b in zip(dates+labels, (dates+labels)[1:])):
                    raise AssertionError("Input/label dates not strictly increasing")
                window = values[[positions[d] for d in dates]]
                x[i], zeros[i] = raw_window(window)
                future = values[[positions[d] for d in labels], 3]
                if not np.isfinite(future).all() or (future <= 0).any():
                    raise ValueError("Invalid future closes")
                # The existing panel computes targets from float32 closes.
                # Match that numeric definition, then store without further rounding.
                y[i] = future.astype(np.float32)/np.float32(window[-1,3])-np.float32(1)
        offset += len(chunk)
        print(json.dumps({"stage":"cache_windows", "rows":offset, "total":n}), flush=True)
    train_ids = meta.index[meta.split.eq("train") & meta.as_of_date.le(TRAIN_END)].to_numpy()
    center, scale = fit_robust(x, train_ids)
    for begin in range(0, n, 15000):
        end = min(begin+15000,n)
        x[begin:end] = normalize(np.array(x[begin:end]), center, scale, zeros[begin:end])
    x.flush(); y.flush()
    write_json(cache_dir/"normalizer.json", {"center":center.tolist(), "scale":scale.tolist(),
        "fit_rows":len(train_ids), "fit_split":"train", "mad_factor":1.4826, "clip":3})
    write_json(cache_dir/"manifest.json",manifest)
    write_json(cache_dir/"complete.json", {"rows":n, "x_sha256":sha256(cache_dir/"x.npy"),
        "y_sha256":sha256(cache_dir/"y.npy"), "meta_sha256":sha256(cache_dir/"meta.csv")})
    return load_cache(cache_dir)


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    return (np.load(cache_dir/"x.npy", mmap_mode="r"), np.load(cache_dir/"y.npy", mmap_mode="r"),
            pd.read_csv(cache_dir/"meta.csv", dtype={"stock":str, "as_of_date":str,
                        "label_start_date":str, "label_end_date":str}))


def make_tasks(meta, split, *, step=20):
    """Select support using labels known strictly before each query block."""
    dates = sorted(meta.loc[meta.split.eq(split), "as_of_date"].unique())
    by_date = {date:group.row_id.to_numpy(dtype=int) for date,group in meta.groupby("as_of_date",sort=True)}
    latest_label = meta.groupby("as_of_date").label_end_date.max().to_dict()
    tasks = []
    for start in range(0,len(dates),step):
        query_dates = dates[start:start+step]
        # Entire eligible cross-section for a support date must be mature.
        support_dates = [d for d in by_date if d<query_dates[0] and latest_label[d]<query_dates[0]][-step:]
        if len(support_dates) < step:
            if split == "train":
                continue
            raise ValueError("Insufficient complete mature support dates")
        support = np.concatenate([by_date[d] for d in support_dates])
        query = np.concatenate([by_date[d] for d in query_dates])
        mature = query[meta.iloc[query].label_end_date.le(query_dates[-1]).to_numpy()]
        tasks.append({"support":support, "query":query, "mature":mature,
                      "begin":query_dates[0], "end":query_dates[-1]})
    return tasks


def prediction_frame(meta, ids, prediction, labels, seed):
    out = meta.iloc[ids][["as_of_date","stock","label_start_date","label_end_date"]].copy()
    for k in range(10):
        out[f"pred_return_{k+1}"] = prediction[:,k]
        out[f"actual_return_{k+1}"] = np.asarray(labels[ids,k])
    out["pred_signal"] = prediction.mean(axis=1)
    out["actual_signal"] = np.asarray(labels[ids]).mean(axis=1)
    out["pred_endpoint_return"] = prediction[:,-1]
    out["actual_endpoint_return"] = np.asarray(labels[ids,-1])
    out["seed"], out["method"] = seed, "DoubleAdapt_GRU_90x10"
    return out
