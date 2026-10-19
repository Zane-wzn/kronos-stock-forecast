"""Normalize saved artifacts without inventing missing targets or model outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for target, aliases in {
        "as_of_date": ("prediction_date", "date", "signal_date"),
        "stock": ("stock_id", "symbol", "code"),
        "pred_signal": ("pred_path_mean_return",),
        "actual_signal": ("actual_path_mean_return",),
        "actual_return_1": ("daily_return_1",),
    }.items():
        if target not in data:
            for alias in aliases:
                if alias in data:
                    data[target] = data[alias]
                    break
    # Return trajectories use the signal-date close as their common baseline.
    if "current_close" in data:
        close = pd.to_numeric(data.current_close, errors="coerce").where(lambda s: s > 0)
        for horizon in range(1, 11):
            for prefix in ("pred", "actual"):
                target, source = f"{prefix}_return_{horizon}", f"{prefix}_close_{horizon}"
                if target not in data and source in data:
                    data[target] = pd.to_numeric(data[source], errors="coerce") / close - 1
    for prefix in ("pred", "actual"):
        path = [f"{prefix}_return_{h}" for h in range(1, 11)]
        target = f"{prefix}_signal"
        if target not in data and all(column in data for column in path):
            values = data[path].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            data[target] = values.mean(axis=1, skipna=False)
        endpoint = f"{prefix}_endpoint_return"
        if endpoint not in data and path[-1] in data:
            data[endpoint] = data[path[-1]]
    return data
