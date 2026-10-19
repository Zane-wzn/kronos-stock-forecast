from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .inputs import normalize_predictions


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str = "CSI300"
    test_start: str | None = "2025-07-01"
    test_end: str | None = "2026-06-05"
    top_k: int | None = None
    top_k_fraction: float | None = None
    drop_n: int | None = None
    min_holding_days: int = 5
    cost_rate: float | None = None
    annualization_days: int | None = None
    bootstrap_reps: int = 5000
    bootstrap_block: int = 10
    seed: int = 100


MARKET_DEFAULTS = {
    "CSI300": {"top_k": 30, "top_k_fraction": None, "drop_n": 3, "cost_rate": 0.0015},
    "CSI800": {"top_k": 200, "top_k_fraction": None, "drop_n": 10, "cost_rate": 0.0015},
    "S&P500": {"top_k": 50, "top_k_fraction": None, "drop_n": 5, "cost_rate": 0.0015},
    "Yahoo S&P 500": {"top_k": 50, "top_k_fraction": None, "drop_n": 5, "cost_rate": 0.0015},
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _newey_west_t(values: np.ndarray, lag: int = 9) -> float:
    x = values[np.isfinite(values)]
    n = len(x)
    if n < 2:
        return float("nan")
    residual = x - x.mean()
    variance = float(np.dot(residual, residual) / n)
    max_lag = min(lag, n - 1)
    for offset in range(1, max_lag + 1):
        gamma = float(np.dot(residual[offset:], residual[:-offset]) / n)
        variance += 2 * (1 - offset / (max_lag + 1)) * gamma
    se = math.sqrt(max(variance / n, 0.0))
    return float(x.mean() / se) if se else float("nan")


def _bootstrap_ci(values: np.ndarray, *, reps: int, block: int, seed: int) -> list[float] | None:
    x = values[np.isfinite(values)]
    n = len(x)
    if not n or reps < 1:
        return None
    width = min(block, n)
    blocks = np.lib.stride_tricks.sliding_window_view(x, width)
    count = math.ceil(n / width)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(blocks), size=(reps, count))
    draw = blocks[picks].reshape(reps, -1)[:, :n]
    return [float(v) for v in np.quantile(draw.mean(axis=1), [0.025, 0.975])]


def _point_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    p, y = pred[mask], actual[mask]
    if not len(p):
        return {"status": "无法验证", "reason": "没有成对的有限预测值和真实值", "n": 0}
    error = p - y
    mae = float(np.mean(np.abs(error)))
    baseline_mae = float(np.mean(np.abs(y)))
    baseline_sse = float(np.sum(y**2))
    return {
        "status": "可验证",
        "scale": "return",
        "n": int(len(p)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mase_vs_zero_return": mae / baseline_mae if baseline_mae else None,
        "oos_r2_vs_zero_return": 1 - float(np.sum(error**2)) / baseline_sse if baseline_sse else None,
    }


def _direction_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    p, y = pred[mask] > 0, actual[mask] > 0
    if not len(p):
        return {"status": "无法验证", "reason": "没有成对的有限预测值和真实值", "n": 0}
    tp, tn = int(np.sum(p & y)), int(np.sum(~p & ~y))
    fp, fn = int(np.sum(p & ~y)), int(np.sum(~p & y))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "status": "可验证", "n": int(len(p)), "accuracy": (tp + tn) / len(p),
        "balanced_accuracy": float(np.nanmean([recall, specificity])), "precision": precision,
        "recall": recall, "f1": f1, "specificity": specificity,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "positive_definition": "return > 0; zero return is negative",
    }


def _cross_section(frame: pd.DataFrame, pred_col: str, actual_col: str, config: EvaluationConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for date, group in frame.groupby("as_of_date", sort=True):
        pred, actual = _numeric(group, pred_col), _numeric(group, actual_col)
        mask = np.isfinite(pred) & np.isfinite(actual)
        pred, actual = pred[mask], actual[mask]
        rows.append({"as_of_date": str(date), "n": int(len(pred)), "ic": _correlation(pred, actual),
                     "rankic": _correlation(pd.Series(pred).rank(method="average").to_numpy(),
                                            pd.Series(actual).rank(method="average").to_numpy())})
    summary: dict[str, Any] = {}
    for offset, name in enumerate(("ic", "rankic")):
        values = np.asarray([row[name] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        mean = float(np.mean(values)) if len(values) else float("nan")
        summary[name] = {
            "count": int(len(values)), "mean": mean if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "positive_ratio": float(np.mean(values > 0)) if len(values) else None,
            "ir_mean_over_std": mean / std if std > 0 else None,
            "annualized_ir_sqrt252": mean / std * math.sqrt(252) if std > 0 else None,
            "newey_west_lag9_mean_tstat": _newey_west_t(values),
            "moving_block_bootstrap_ci95": _bootstrap_ci(
                values, reps=config.bootstrap_reps, block=config.bootstrap_block, seed=config.seed + offset
            ),
            "bootstrap_block": config.bootstrap_block,
            "bootstrap_reps": config.bootstrap_reps,
        }
    return summary, rows


def _portfolio(frame: pd.DataFrame, signal_col: str, actual_col: str, *, top_k: int | None,
               top_k_fraction: float | None, drop_n: int, min_holding_days: int,
               cost_rate: float, annualization_days: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    previous: dict[str, float] = {}
    age: dict[str, int] = {}
    rows = []
    if "actual_return_1" not in frame or not np.isfinite(_numeric(frame, "actual_return_1")).any():
        return {"status": "无法验证", "reason": "缺少下一交易日 actual_return_1；不能用10日 Path/Endpoint 标签代替日收益"}, []
    for day_index, (date, source) in enumerate(frame.groupby("as_of_date", sort=True)):
        part = pd.DataFrame({"stock": source["stock"].astype(str), "signal": _numeric(source, signal_col),
                             "actual": _numeric(source, actual_col), "daily_return": _numeric(source, "actual_return_1")})
        part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["signal", "daily_return"])
        part = part.sort_values(["signal", "stock"], ascending=[False, True], kind="mergesort")
        n = len(part)
        if not n:
            continue
        k = min(n, max(1, math.ceil(n * top_k_fraction)) if top_k_fraction is not None else (top_k or 1))
        ranked = part.stock.tolist()
        desired = set(ranked[:k])
        universe = set(ranked)
        forced = set(previous) - universe
        locked = {stock for stock in previous if stock in universe and stock not in desired
                  and day_index - age.get(stock, day_index) < min_holding_days}
        eligible = [stock for stock in previous if stock in universe and stock not in desired and stock not in locked]
        score = dict(zip(part.stock, part.signal))
        eligible.sort(key=lambda stock: (score[stock], stock))
        sell = forced | set(eligible[:drop_n])
        held = set(previous) - sell
        if not previous:
            held = set(ranked[:k])
        elif len(held) < k:
            for stock in ranked[:k]:
                if len(held) >= k:
                    break
                held.add(stock)
        if len(held) > k:
            held = set(sorted(held, key=lambda stock: (-score[stock], stock))[:k])
        weights = {stock: 1 / len(held) for stock in held} if held else {}
        names = set(previous) | set(weights)
        turnover = 1.0 if not previous else 0.5 * sum(abs(weights.get(s, 0) - previous.get(s, 0)) for s in names)
        selected = part[part.stock.isin(held)]
        top_return = float(selected.daily_return.mean())
        age = {stock: (age[stock] if stock in age else day_index) for stock in held}
        bought, sold = held - set(previous), set(previous) - held
        previous = weights
        # Label spreads use the matching multi-step target, whereas the held portfolio
        # consumes only next-day returns. These are distinct fields in the dashboard.
        labels = part.dropna(subset=["actual"])
        label_k = min(k, len(labels))
        label_q = max(1, len(labels) // 5)
        bottom_return = float(labels.tail(label_k).actual.mean()) if label_k else float("nan")
        label_top = float(labels.head(label_k).actual.mean()) if label_k else float("nan")
        label_benchmark = float(labels.actual.mean())
        q5q1 = float(labels.head(label_q).actual.mean() - labels.tail(label_q).actual.mean())
        rows.append({
            "as_of_date": str(date), "n": int(n), "top_k_effective": int(len(held)), "turnover": turnover,
            "bought": len(bought), "sold": len(sold), "forced_exits": len(forced),
            "q5_q1": q5q1, "top_raw": top_return,
            "portfolio_gross_daily_return": top_return,
            "benchmark_daily_return": float(part.daily_return.mean()),
            "bottom_return": bottom_return, "top_excess": label_top - label_benchmark,
            "long_short": label_top - bottom_return, "transaction_cost": cost_rate * turnover,
            "top_net": top_return - cost_rate * turnover,
        })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"status": "无法验证", "reason": "无法形成逐日组合"}, rows
    net = daily.top_net.to_numpy(dtype=float)
    std = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
    wealth = np.cumprod(1 + net)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = wealth / peaks[1:] - 1
    return {
        "status": "可验证", "dates": int(len(daily)), "top_k": top_k, "top_k_fraction": top_k_fraction,
        "drop_n": drop_n, "min_holding_days": min_holding_days,
        "cost_rate": cost_rate, "turnover": float(daily.turnover.mean()),
        "q5_q1": float(daily.q5_q1.mean()), "top_raw": float(daily.top_raw.mean()),
        "mean_portfolio_gross_daily_return": float(daily.portfolio_gross_daily_return.mean()),
        "bottom_return": float(daily.bottom_return.mean()), "top_excess": float(daily.top_excess.mean()),
        "top_net": float(daily.top_net.mean()), "long_short": float(daily.long_short.mean()),
        "sharpe": float(np.mean(net) / std * math.sqrt(annualization_days)) if std > 0 else None,
        "max_drawdown": float(np.min(drawdown)), "net_sharpe": float(np.mean(net) / std * math.sqrt(annualization_days)) if std > 0 else None,
        "net_max_drawdown": float(np.min(drawdown)),
        "annualized_excess_proxy": float((daily.top_net - daily.benchmark_daily_return).mean() * annualization_days),
        "annualized_excess_proxy_definition": "annualization_days * mean(net_daily_return - benchmark_daily_return); CSI300 historical dashboard field kronos_aer uses this proxy",
        "definition": "Q5-Q1/bottom/top excess/long-short use the matching multi-step actual_signal or actual_endpoint_return; held Top-K gross/net daily return uses actual_return_1; equal-weight Drop-N/minimum-holding policy; missing-universe holdings exit forcibly; cost_rate * one-way weight turnover; next-day close-to-close proxy; not the separate Kronos AER execution simulation",
    }, rows


def evaluate_frame(frame: pd.DataFrame, config: EvaluationConfig | None = None) -> dict[str, Any]:
    config = config or EvaluationConfig()
    frame = normalize_predictions(frame)
    if config.dataset not in MARKET_DEFAULTS:
        raise ValueError(f"unknown dataset: {config.dataset}")
    defaults = MARKET_DEFAULTS[config.dataset]
    top_k = config.top_k if config.top_k is not None else defaults["top_k"]
    top_k_fraction = config.top_k_fraction if config.top_k_fraction is not None else defaults["top_k_fraction"]
    drop_n = config.drop_n if config.drop_n is not None else defaults["drop_n"]
    cost_rate = config.cost_rate if config.cost_rate is not None else defaults["cost_rate"]
    annualization_days = config.annualization_days if config.annualization_days is not None else (238 if config.dataset == "CSI800" else 252)
    if top_k < 1 or drop_n < 0 or config.min_holding_days < 1 or cost_rate < 0 or annualization_days < 1:
        raise ValueError("invalid portfolio configuration")
    if top_k_fraction is not None and not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if config.bootstrap_reps < 0 or config.bootstrap_block < 1:
        raise ValueError("invalid bootstrap configuration")
    missing = {"as_of_date", "stock"} - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file missing required columns: {sorted(missing)}")
    data = frame.copy()
    if config.test_start and config.test_end and pd.Timestamp(config.test_start) > pd.Timestamp(config.test_end):
        raise ValueError("test_start must be before test_end")
    data["as_of_date"] = pd.to_datetime(data["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if data["as_of_date"].isna().any() or data.stock.isna().any():
        raise ValueError("missing/invalid prediction date or stock ID")
    data["stock"] = data.stock.astype(str).str.strip()
    if data.stock.eq("").any():
        raise ValueError("empty stock ID")
    if config.test_start:
        data = data[data.as_of_date >= config.test_start]
    if config.test_end:
        data = data[data.as_of_date <= config.test_end]
    if data.duplicated(["as_of_date", "stock"]).any():
        raise ValueError("duplicate (as_of_date, stock) rows; resolve alignment before evaluation")
    if data.empty:
        raise ValueError("no prediction rows inside the selected test window")

    result: dict[str, Any] = {
        "source_metadata": {column: sorted(str(v) for v in data[column].dropna().unique())
                            for column in ("group", "seed", "method", "model", "dataset") if column in data},
        "config": {**asdict(config), "top_k": top_k, "top_k_fraction": top_k_fraction,
                   "drop_n": drop_n,
                   "annualization_days": annualization_days,
                   "cost_rate": cost_rate, "observed_dates": int(data.as_of_date.nunique()),
                   "observed_rows": int(len(data))},
        "prediction_files_expected_columns": ["as_of_date", "stock", "pred_signal", "actual_signal",
                                               "pred_return_1..10", "actual_return_1..10",
                                               "pred_endpoint_return", "actual_endpoint_return"],
        "metrics": {},
    }
    for mode, pred_col, actual_col in (("path", "pred_signal", "actual_signal"),
                                       ("endpoint", "pred_endpoint_return", "actual_endpoint_return")):
        if pred_col not in data or actual_col not in data:
            unavailable = {"status": "无法验证", "reason": f"缺少独立的 {mode} 预测或标签列"}
            result["metrics"][mode] = {"point": unavailable, "direction": unavailable, "cross_section": unavailable, "portfolio": unavailable}
            result.setdefault("daily", {})[mode] = {"cross_section": [], "portfolio": []}
            continue
        pred, actual = _numeric(data, pred_col), _numeric(data, actual_col)
        point = _point_metrics(pred, actual)
        direction = _direction_metrics(pred, actual)
        cross, cross_daily = _cross_section(data, pred_col, actual_col, config)
        portfolio, portfolio_daily = _portfolio(data, pred_col, actual_col, top_k=top_k,
                                                  top_k_fraction=top_k_fraction, drop_n=drop_n,
                                                  min_holding_days=config.min_holding_days, cost_rate=cost_rate,
                                                  annualization_days=annualization_days)
        result["metrics"][mode] = {"point": point, "direction": direction,
                                   "cross_section": cross, "portfolio": portfolio}
        result.setdefault("daily", {})[mode] = {
            "cross_section": cross_daily, "portfolio": portfolio_daily,
        }

    horizons = []
    pred_trajectory, actual_trajectory = [], []
    for horizon in range(1, 11):
        p_col, a_col = f"pred_return_{horizon}", f"actual_return_{horizon}"
        if p_col in data and a_col in data:
            p, a = _numeric(data, p_col), _numeric(data, a_col)
            horizons.append({"horizon": horizon, **_point_metrics(p, a)})
            pred_trajectory.append(p)
            actual_trajectory.append(a)
        else:
            horizons.append({"horizon": horizon, "status": "无法验证", "reason": f"缺少 {p_col} 或 {a_col}"})
    if len(pred_trajectory) == 10:
        pooled = _point_metrics(np.concatenate(pred_trajectory), np.concatenate(actual_trajectory))
        pooled["scale"] = "10-step cumulative return trajectory pooled across horizons"
        result["metrics"]["trajectory_pooled"] = pooled
    else:
        result["metrics"]["trajectory_pooled"] = {"status": "无法验证", "reason": "需要完整保存 horizon 1 至 10 预测和标签"}
    result["metrics"]["trajectory_by_horizon"] = horizons
    result["metrics"]["kronos_aer"] = {
        "status": "无法验证", "reason": "需要独立 Kronos 开盘执行回测的逐日报告；普通 Top-K 日收益不替代 AER",
        "definition": "238 * mean(return - benchmark - cost); benchmark is test-universe equal-weight proxy",
    }
    result["metrics"]["validation_rankic_gate"] = {
        "status": "无法验证", "reason": "需读取实验的验证/更新日志；验证门不属于测试指标",
    }
    return clean_json(result)
