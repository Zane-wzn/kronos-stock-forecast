"""Build and evaluate a dynamic two-branch CSI300 prediction ensemble.

This script does not train a model.  It combines the existing static Adapter
and continual Adapter prediction artifacts, updates their Hedge weights only
after month-end predictions, and evaluates all requested baselines with the
same signal and portfolio protocol.

The prediction artifacts and the portfolio-return label artifact can be
different files.  This is required for head-only experiments whose prediction
files contain ``actual_signal`` but do not carry ``actual_return_1``.

The implementation is deliberately independent from the original ABC runner:
existing artifacts are never overwritten and a new output directory is used
for the dynamic experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_abcs.portfolio import PortfolioConfig, backtest_long_only


DEFAULT_START = "2025-07-01"
DEFAULT_END = "2026-06-05"
METHODS = ("A", "B", "C", "Static", "Continual", "C_dynamic")


def validate_initial_alpha(initial_alpha: float, alpha_min: float, alpha_max: float) -> None:
    if not (0.0 < alpha_min <= initial_alpha <= alpha_max < 1.0):
        raise ValueError("initial_alpha must lie within 0 < alpha_min <= alpha <= alpha_max < 1")


def mix_normalized_signals(static_signal, continual_signal, alpha: float) -> np.ndarray:
    static = np.asarray(static_signal, dtype=float)
    continual = np.asarray(continual_signal, dtype=float)
    if static.shape != continual.shape:
        raise ValueError("static and continual normalized signals must have matching shapes")
    if not np.isfinite(static).all() or not np.isfinite(continual).all() or not math.isfinite(alpha):
        raise ValueError("signals and alpha must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return (1.0 - alpha) * static + alpha * continual


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _prepare_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"as_of_date", "stock", "pred_signal", "actual_signal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="raise").dt.normalize()
    out["stock"] = out["stock"].astype(str).str.strip()
    for col in ("pred_signal", "actual_signal"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["as_of_date", "stock"]].duplicated().any():
        raise ValueError(f"duplicate (as_of_date, stock) keys in {path}")
    if out["pred_signal"].isna().any() or out["actual_signal"].isna().any():
        raise ValueError(f"non-finite prediction or target values in {path}")
    return out.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)


def _assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    left_keys = set(map(tuple, left[["as_of_date", "stock"]].astype(str).to_numpy()))
    right_keys = set(map(tuple, right[["as_of_date", "stock"]].astype(str).to_numpy()))
    missing = sorted(left_keys - right_keys)
    extra = sorted(right_keys - left_keys)
    if missing or extra:
        raise ValueError(
            f"prediction key mismatch between {left_name} and {right_name}: "
            f"missing_in_{right_name}={missing[:5]}, extra_in_{right_name}={extra[:5]}"
        )


def _load_return_reference(path: Path) -> pd.DataFrame:
    """Load only canonical realized returns; never use its prediction columns."""
    frame = pd.read_csv(path)
    required = {"as_of_date", "stock", "actual_return_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing return-label columns: {sorted(missing)}")
    out = frame[["as_of_date", "stock", "actual_return_1"]].copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="raise").dt.normalize()
    out["stock"] = out["stock"].astype(str).str.strip()
    out["actual_return_1"] = pd.to_numeric(out["actual_return_1"], errors="coerce")
    if out[["as_of_date", "stock"]].duplicated().any():
        raise ValueError(f"duplicate (as_of_date, stock) keys in return reference {path}")
    if not np.isfinite(out["actual_return_1"].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite actual_return_1 values in return reference {path}")
    return out.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)


def _month_end_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).normalize().unique()))
    return {
        dates[i]
        for i in range(len(dates) - 1)
        if dates[i].month != dates[i + 1].month
    }


def rank_gaussian(values: Iterable[float]) -> np.ndarray:
    """Map cross-sectional ranks to a standard-normal-like scale."""
    x = pd.Series(values, dtype=float)
    if x.empty:
        return np.asarray([], dtype=float)
    ranks = x.rank(method="average").to_numpy(dtype=float)
    n = len(ranks)
    normal = NormalDist()
    return np.asarray(
        [normal.inv_cdf(float((rank - 0.5) / n)) for rank in ranks],
        dtype=float,
    )


def add_daily_rank_gaussian(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """Add a date-wise Rank-Gaussian column without using target labels."""
    if source not in frame.columns:
        raise ValueError(f"missing source column {source}")
    out = frame.copy()
    out[target] = np.nan
    for _, group in out.groupby("as_of_date", sort=False):
        out.loc[group.index, target] = rank_gaussian(group[source].to_numpy(dtype=float))
    return out


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(x[mask].corr(y[mask], method=method))


def _daily_signal_metric(group: pd.DataFrame, score_col: str) -> dict[str, Any]:
    g = group[[score_col, "actual_signal", "stock"]].dropna().copy()
    if len(g) < 3:
        return {"n_stocks": int(len(g)), "rankic": np.nan, "pearson_ic": np.nan, "q5_q1": np.nan}
    q = max(1, len(g) // 5)
    ordered = g.sort_values([score_col, "stock"], kind="mergesort")
    return {
        "n_stocks": int(len(g)),
        "rankic": _corr(g[score_col], g["actual_signal"], "spearman"),
        "pearson_ic": _corr(g[score_col], g["actual_signal"], "pearson"),
        "q5_q1": float(ordered.tail(q)["actual_signal"].mean() - ordered.head(q)["actual_signal"].mean()),
    }


def daily_signal_metrics(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("as_of_date", sort=True):
        row = {"as_of_date": date, **_daily_signal_metric(group, score_col)}
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_loss_metrics(group: pd.DataFrame, pred_col: str) -> dict[str, float]:
    g = group[[pred_col, "actual_signal"]].dropna().copy()
    if len(g) < 3:
        return {"rankic": np.nan, "standardized_mae": np.nan}
    pred = g[pred_col].to_numpy(dtype=float)
    actual = g["actual_signal"].to_numpy(dtype=float)
    pred_std = float(pred.std(ddof=0))
    actual_std = float(actual.std(ddof=0))
    if pred_std <= 1e-12 or actual_std <= 1e-12:
        return {"rankic": np.nan, "standardized_mae": np.nan}
    return {
        "rankic": _corr(g[pred_col], g["actual_signal"], "spearman"),
        "standardized_mae": float(
            np.mean(np.abs((pred - pred.mean()) / pred_std - (actual - actual.mean()) / actual_std))
        ),
    }


def model_loss(frame: pd.DataFrame, pred_col: str, rank_weight: float = 0.7, error_weight: float = 0.3) -> dict[str, Any]:
    """Calculate the loss used by Hedge from already mature daily labels."""
    daily: list[dict[str, Any]] = []
    for date, group in frame.groupby("as_of_date", sort=True):
        metrics = _daily_loss_metrics(group, pred_col)
        metrics["as_of_date"] = date
        daily.append(metrics)
    metrics_frame = pd.DataFrame(daily, columns=["rankic", "standardized_mae", "as_of_date"])
    valid = metrics_frame.dropna(subset=["rankic", "standardized_mae"])
    if valid.empty:
        return {
            "loss": np.nan,
            "rankic_mean": np.nan,
            "standardized_mae_mean": np.nan,
            "rank_days": 0,
            "error_days": 0,
            "daily": metrics_frame,
        }
    rankic_mean = float(valid["rankic"].mean())
    error_mean = float(valid["standardized_mae"].mean())
    error_loss = float(np.clip(error_mean / 2.0, 0.0, 1.0))
    loss = float(rank_weight * (1.0 - rankic_mean) + error_weight * error_loss)
    return {
        "loss": loss,
        "rankic_mean": rankic_mean,
        "standardized_mae_mean": error_mean,
        "error_loss": error_loss,
        "rank_days": int(valid["rankic"].notna().sum()),
        "error_days": int(valid["standardized_mae"].notna().sum()),
        "daily": metrics_frame,
    }


def update_hedge_weights(
    static_weight: float,
    continual_weight: float,
    static_loss: float,
    continual_loss: float,
    *,
    eta: float = 0.5,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
) -> dict[str, float]:
    if not (0.0 < alpha_min <= alpha_max < 1.0):
        raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max < 1")
    if eta <= 0 or static_weight <= 0 or continual_weight <= 0:
        raise ValueError("Hedge weights and eta must be positive")
    if not np.isfinite(static_loss) or not np.isfinite(continual_loss):
        return {
            "static_weight": float(static_weight),
            "continual_weight": float(continual_weight),
            "raw_alpha": float(continual_weight / (static_weight + continual_weight)),
            "alpha": float(np.clip(continual_weight / (static_weight + continual_weight), alpha_min, alpha_max)),
        }
    new_static = float(static_weight * np.exp(-eta * static_loss))
    new_continual = float(continual_weight * np.exp(-eta * continual_loss))
    raw_alpha = new_continual / (new_static + new_continual)
    return {
        "static_weight": new_static,
        "continual_weight": new_continual,
        "raw_alpha": float(raw_alpha),
        "alpha": float(np.clip(raw_alpha, alpha_min, alpha_max)),
    }


def _load_and_join(static_path: Path, continual_path: Path) -> pd.DataFrame:
    static = _prepare_frame(pd.read_csv(static_path), static_path)
    continual = _prepare_frame(pd.read_csv(continual_path), continual_path)
    _assert_same_keys(static, continual, "static", "continual")

    target_comparison = static[["as_of_date", "stock", "actual_signal"]].merge(
        continual[["as_of_date", "stock", "actual_signal"]],
        on=["as_of_date", "stock"],
        how="inner",
        validate="one_to_one",
        suffixes=("_static", "_continual"),
    )
    if not np.allclose(
        target_comparison["actual_signal_static"].to_numpy(dtype=float),
        target_comparison["actual_signal_continual"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("static and continual artifacts do not contain identical actual_signal labels")

    static = static.rename(columns={"pred_signal": "static_raw_signal"})
    continual = continual.rename(columns={"pred_signal": "continual_raw_signal"})
    keep_continual = ["as_of_date", "stock", "continual_raw_signal"]
    for col in ("label_end_date", "model_version", "update_ran_after_prediction"):
        if col in continual.columns:
            keep_continual.append(col)
    # Keep the maturity metadata from the continual artifact only.  Both
    # prediction files commonly contain label_end_date; retaining both would
    # make pandas suffix the column and break the explicit maturity check.
    static = static[["as_of_date", "stock", "static_raw_signal", "actual_signal"]]
    merged = static.merge(
        continual[keep_continual],
        on=["as_of_date", "stock"],
        how="inner",
        validate="one_to_one",
    )
    if "label_end_date" not in merged.columns:
        raise ValueError("continual predictions must contain label_end_date for maturity checks")
    merged["label_end_date"] = pd.to_datetime(merged["label_end_date"], errors="raise").dt.normalize()
    merged["static_raw_signal"] = pd.to_numeric(merged["static_raw_signal"], errors="coerce")
    merged["continual_raw_signal"] = pd.to_numeric(merged["continual_raw_signal"], errors="coerce")
    if not np.isfinite(merged[["static_raw_signal", "continual_raw_signal"]].to_numpy(dtype=float)).all():
        raise ValueError("joined prediction signals contain non-finite values")
    return merged.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)


def build_dynamic_predictions(
    joined: pd.DataFrame,
    *,
    test_start: str = DEFAULT_START,
    test_end: str = DEFAULT_END,
    history_days: int = 60,
    eta: float = 0.5,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    rank_weight: float = 0.7,
    error_weight: float = 0.3,
    initial_alpha: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions and an auditable month-end weight log."""
    if history_days < 1 or not math.isclose(rank_weight + error_weight, 1.0, abs_tol=1e-12):
        raise ValueError("history_days must be positive and loss weights must sum to 1")
    validate_initial_alpha(initial_alpha, alpha_min, alpha_max)
    start = pd.Timestamp(test_start).normalize()
    end = pd.Timestamp(test_end).normalize()
    data = joined[(joined["as_of_date"] >= start) & (joined["as_of_date"] <= end)].copy()
    if data.empty:
        raise ValueError("no rows in requested test interval")
    data = add_daily_rank_gaussian(data, "static_raw_signal", "static_normalized_signal")
    data = add_daily_rank_gaussian(data, "continual_raw_signal", "continual_normalized_signal")

    dates = pd.DatetimeIndex(sorted(data["as_of_date"].unique()))
    month_ends = _month_end_dates(dates)
    static_weight, continual_weight = 1.0 - initial_alpha, initial_alpha
    alpha = initial_alpha
    prediction_rows: list[pd.DataFrame] = []
    update_rows: list[dict[str, Any]] = []

    for position, date in enumerate(dates):
        day = data[data["as_of_date"].eq(date)].copy()
        day["alpha"] = alpha
        day["final_signal"] = mix_normalized_signals(
            day["static_normalized_signal"], day["continual_normalized_signal"], alpha)
        prediction_rows.append(day)

        if date not in month_ends:
            continue

        mature = data[(data["as_of_date"] < date) & (data["label_end_date"] <= date)].copy()
        mature_dates = pd.DatetimeIndex(sorted(mature["as_of_date"].unique()))
        mature_dates = mature_dates[-history_days:]
        window = mature[mature["as_of_date"].isin(mature_dates)].copy()
        static_result = model_loss(window, "static_raw_signal", rank_weight, error_weight)
        continual_result = model_loss(window, "continual_raw_signal", rank_weight, error_weight)
        before_alpha = alpha
        static_weight_before = static_weight
        continual_weight_before = continual_weight
        update = update_hedge_weights(
            static_weight,
            continual_weight,
            static_result["loss"],
            continual_result["loss"],
            eta=eta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        static_weight = update["static_weight"]
        continual_weight = update["continual_weight"]
        alpha = update["alpha"]
        update_rows.append(
            {
                "update_date": date,
                "next_test_date": dates[position + 1] if position + 1 < len(dates) else pd.NaT,
                "window_start": mature_dates.min() if len(mature_dates) else pd.NaT,
                "window_end_exclusive": date,
                "matured_rows": int(len(window)),
                "matured_dates": int(len(mature_dates)),
                "alpha_before": before_alpha,
                "static_weight_before": static_weight_before,
                "continual_weight_before": continual_weight_before,
                "static_rankic_mean": static_result["rankic_mean"],
                "continual_rankic_mean": continual_result["rankic_mean"],
                "static_standardized_mae_mean": static_result["standardized_mae_mean"],
                "continual_standardized_mae_mean": continual_result["standardized_mae_mean"],
                "static_loss": static_result["loss"],
                "continual_loss": continual_result["loss"],
                "static_weight_after": static_weight,
                "continual_weight_after": continual_weight,
                "raw_alpha_after": update["raw_alpha"],
                "alpha_after": alpha,
                "update_applied": bool(np.isfinite(static_result["loss"]) and np.isfinite(continual_result["loss"])),
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions = predictions.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)
    updates = pd.DataFrame(update_rows)
    return predictions, updates


def build_fixed_weight_predictions(
    joined: pd.DataFrame,
    *,
    test_start: str = DEFAULT_START,
    test_end: str = DEFAULT_END,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """Mix the same static/continual streams with a fixed continual weight.

    This is intentionally a pure replay of already generated prediction
    streams. It performs no fitting and no access to labels beyond retaining
    the labels carried by the joined artifact, so it is the paired control for
    the dynamic Hedge policy.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("fixed alpha must be finite and between 0 and 1")
    start = pd.Timestamp(test_start).normalize()
    end = pd.Timestamp(test_end).normalize()
    data = joined[(joined["as_of_date"] >= start) & (joined["as_of_date"] <= end)].copy()
    if data.empty:
        raise ValueError("no rows in requested test interval")
    data = add_daily_rank_gaussian(data, "static_raw_signal", "static_normalized_signal")
    data = add_daily_rank_gaussian(data, "continual_raw_signal", "continual_normalized_signal")
    data["alpha"] = float(alpha)
    data["final_signal"] = mix_normalized_signals(
        data["static_normalized_signal"], data["continual_normalized_signal"], float(alpha)
    )
    return data.sort_values(["as_of_date", "stock"], kind="mergesort").reset_index(drop=True)


def _read_baseline(path: Path, method: str) -> pd.DataFrame:
    frame = _prepare_frame(pd.read_csv(path), path)
    return frame[["as_of_date", "stock", "pred_signal", "actual_signal"]].rename(
        columns={"pred_signal": "prediction_score", "actual_signal": "actual_signal"}
    ).assign(method=method)


def _signal_summary(frame: pd.DataFrame, score_col: str = "prediction_score") -> dict[str, Any]:
    daily = daily_signal_metrics(frame, score_col)
    return {
        "dates": int(len(daily)),
        "rows": int(len(frame)),
        "mean_rankic": _finite(daily["rankic"].mean()) if not daily.empty else None,
        "median_rankic": _finite(daily["rankic"].median()) if not daily.empty else None,
        "rankic_std": _finite(daily["rankic"].std(ddof=1)) if len(daily) > 1 else None,
        "rankic_positive_ratio": _finite((daily["rankic"] > 0).mean()) if not daily.empty else None,
        "mean_pearson_ic": _finite(daily["pearson_ic"].mean()) if not daily.empty else None,
        "mean_q5_q1": _finite(daily["q5_q1"].mean()) if not daily.empty else None,
    }


def _backtest(frame: pd.DataFrame, method: str, portfolio_config: PortfolioConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"as_of_date", "stock", "prediction_score", "actual_return_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{method} missing backtest columns: {sorted(missing)}")
    data = frame[["as_of_date", "stock", "prediction_score", "actual_return_1"]].rename(
        columns={"actual_return_1": "next_day_return"}
    )
    result = backtest_long_only(
        data,
        score_col="prediction_score",
        return_col="next_day_return",
        config=portfolio_config,
        dataset="CSI300",
        model=method,
        seed=100,
    )
    return dict(result["summary"]), result["daily"].copy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.experiment_root)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite to replace artifacts")
    output.mkdir(parents=True, exist_ok=True)

    static_path = Path(args.static_predictions)
    continual_path = Path(args.continual_predictions)
    joined = _load_and_join(static_path, continual_path)
    dynamic, weight_log = build_dynamic_predictions(
        joined,
        test_start=args.test_start,
        test_end=args.test_end,
        history_days=args.history_days,
        eta=args.eta,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        rank_weight=args.rank_weight,
        error_weight=args.error_weight,
        initial_alpha=args.initial_alpha,
    )

    dynamic_out = dynamic.copy()
    dynamic_out["prediction_score"] = dynamic_out["final_signal"]
    _atomic_csv(output / "predictions.csv", dynamic_out)
    _atomic_csv(output / "weight_log.csv", weight_log)

    static_for_eval = joined.rename(columns={"static_raw_signal": "prediction_score"}).copy()
    continual_for_eval = joined.rename(columns={"continual_raw_signal": "prediction_score"}).copy()
    eval_frames: dict[str, pd.DataFrame] = {
        "Static": static_for_eval,
        "Continual": continual_for_eval,
        "C_dynamic": dynamic_out,
    }
    abc_dir = Path(args.abc_dir)
    for group in ("A", "B", "C"):
        path = abc_dir / f"predictions_{group}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing ABC baseline: {path}")
        baseline = _prepare_frame(pd.read_csv(path), path)
        _assert_same_keys(joined, baseline, "adapter predictions", f"ABC {group}")
        eval_frames[group] = baseline.rename(columns={"pred_signal": "prediction_score"})

    # Attach only the canonical next-day return.  In head-only experiments it
    # comes from a separate label artifact, while all prediction scores above
    # still come exclusively from the requested static/continual files.
    returns_path = Path(args.returns_predictions) if args.returns_predictions else static_path
    if args.returns_predictions:
        actual_returns = _load_return_reference(returns_path)
        _assert_same_keys(joined, actual_returns, "prediction", "return reference")
    else:
        actual_returns = _load_return_reference(static_path)
        _assert_same_keys(joined, actual_returns, "prediction", "static return reference")
    summaries: list[dict[str, Any]] = []
    portfolio_config = PortfolioConfig(
        top_k=args.top_k,
        drop_n=args.drop_n,
        min_holding_days=args.min_holding_days,
        cost_rate=args.cost_rate,
    )
    comparison_daily: list[pd.DataFrame] = []
    for method in METHODS:
        frame = eval_frames[method].merge(
            actual_returns,
            on=["as_of_date", "stock"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_canonical"),
        )
        _atomic_csv(output / f"{method}_daily_signal_metrics.csv", daily_signal_metrics(frame, "prediction_score"))
        signal = _signal_summary(frame)
        portfolio, portfolio_daily = _backtest(frame, method, portfolio_config)
        _atomic_csv(output / f"{method}_daily_returns.csv", portfolio_daily)
        summaries.append({"method": method, **{f"signal_{k}": v for k, v in signal.items()}, **{f"portfolio_{k}": v for k, v in portfolio.items()}})
        comparison_daily.append(portfolio_daily.assign(method=method))

    comparison = pd.DataFrame(summaries)
    _atomic_csv(output / "comparison_summary.csv", comparison)
    all_daily_returns = pd.concat(comparison_daily, ignore_index=True)
    _atomic_csv(output / "all_methods_daily_returns.csv", all_daily_returns)

    dynamic_daily_signal = daily_signal_metrics(dynamic_out, "prediction_score")
    dynamic_daily = dynamic_daily_signal.merge(
        all_daily_returns[all_daily_returns["method"].eq("C_dynamic")],
        on="as_of_date",
        how="left",
        validate="one_to_one",
    )
    dynamic_daily = dynamic_daily.merge(
        dynamic_out[["as_of_date", "alpha"]].drop_duplicates("as_of_date"),
        on="as_of_date",
        how="left",
        validate="one_to_one",
    )
    _atomic_csv(output / "daily_metrics.csv", dynamic_daily)

    config = {
        "status": "complete",
        "experiment": "csi300_continual_c_dynamic",
        "test_start": args.test_start,
        "test_end": args.test_end,
        "methods": list(METHODS),
        "static_predictions": str(static_path.resolve()),
        "continual_predictions": str(continual_path.resolve()),
        "returns_predictions": str(returns_path.resolve()),
        "returns_are_prediction_source": False,
        "abc_dir": str(abc_dir.resolve()),
        "dynamic": {
            "normalization": "daily_cross_section_rank_gaussian",
            "history_days": args.history_days,
            "update_frequency": "month_end_after_prediction",
            "eta": args.eta,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "rank_weight": args.rank_weight,
            "error_weight": args.error_weight,
            "initial_static_weight": 1.0 - args.initial_alpha,
            "initial_continual_weight": args.initial_alpha,
            "loss": "0.7*(1-mean_daily_rankic)+0.3*clip(mean_daily_standardized_mae/2,0,1)",
            "maturity_filter": "as_of_date < update_date and label_end_date <= update_date",
        },
        "portfolio": {
            "top_k": args.top_k,
            "drop_n": args.drop_n,
            "min_holding_days": args.min_holding_days,
            "cost_rate": args.cost_rate,
            "next_day_return": "actual_return_1",
        },
        "rows": int(len(dynamic_out)),
        "dates": int(dynamic_out["as_of_date"].nunique()),
        "updates": int(len(weight_log)),
    }
    _atomic_json(output / "run_config.json", config)
    _atomic_json(
        output / "summary.json",
        {
            "status": "complete",
            "methods": list(METHODS),
            "dynamic_updates": int(len(weight_log)),
            "alpha_mean": _finite(dynamic_out["alpha"].mean()),
            "alpha_min_observed": _finite(dynamic_out["alpha"].min()),
            "alpha_max_observed": _finite(dynamic_out["alpha"].max()),
            "comparison_summary": comparison.to_dict("records"),
        },
    )
    return config


def parse_args() -> argparse.Namespace:
    root = ROOT / "experiments" / "csi300_aligned"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=root)
    parser.add_argument("--static-predictions", type=Path, default=root / "models" / "prediction_adapter" / "predictions.csv")
    parser.add_argument("--continual-predictions", type=Path, default=root / "models" / "adapter_continual" / "predictions.csv")
    parser.add_argument(
        "--returns-predictions",
        type=Path,
        default=None,
        help="Separate artifact containing canonical actual_return_1 labels; its predictions are ignored.",
    )
    parser.add_argument("--abc-dir", type=Path, default=root / "models" / "continual_abc")
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "continual_c_dynamic")
    parser.add_argument("--test-start", default=DEFAULT_START)
    parser.add_argument("--test-end", default=DEFAULT_END)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument("--rank-weight", type=float, default=0.7)
    parser.add_argument("--error-weight", type=float, default=0.3)
    parser.add_argument("--initial-alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--drop-n", type=int, default=3)
    parser.add_argument("--min-holding-days", type=int, default=5)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
