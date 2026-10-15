"""Evaluate D-RankIC Static versus dynamic alpha=0.3-initial continual signal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from scripts.experiments.run_dynamic_continual_csi300 import (
    _atomic_csv, _atomic_json, _assert_same_keys, _load_return_reference,
    _backtest, daily_signal_metrics, mix_normalized_signals,
)
from evaluate_adapter_continual_metrics import (
    portfolio_metrics, newey_west_tstat_mean,
)
from continual_abcs.portfolio import PortfolioConfig


def paired_indices(n, repetitions=5000, block=10, seed=20261013):
    if n < block:
        raise ValueError("at least 10 dates required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block+1, size=(repetitions, math.ceil(n/block)))
    return (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :n]


def evaluate(static_dir: Path, continual_dir: Path, dynamic_dir: Path):
    out = dynamic_dir / "evaluation"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation: {out}")
    out.mkdir(parents=True, exist_ok=True)
    static = pd.read_csv(static_dir / "predictions.csv", dtype={"stock": str})
    continual = pd.read_csv(continual_dir / "predictions.csv", dtype={"stock": str})
    dynamic = pd.read_csv(dynamic_dir / "predictions.csv", dtype={"stock": str})
    for frame in (static, continual, dynamic):
        frame["as_of_date"] = pd.to_datetime(frame.as_of_date).dt.normalize()
        if len(frame) != 66103 or frame.as_of_date.nunique() != 226 or frame[["as_of_date", "stock"]].duplicated().any():
            raise AssertionError("expected unique complete 226-date / 66103-row test panel")
    _assert_same_keys(static, continual, "static", "continual")
    _assert_same_keys(static, dynamic, "static", "dynamic")
    if not np.isfinite(dynamic[["final_signal", "alpha", "static_normalized_signal",
                               "continual_normalized_signal"]].to_numpy()).all():
        raise AssertionError("dynamic outputs contain nonfinite values")
    expected = ((1-dynamic.alpha) * dynamic.static_normalized_signal
                + dynamic.alpha * dynamic.continual_normalized_signal)
    if not np.allclose(expected, dynamic.final_signal, atol=1e-12, rtol=0):
        raise AssertionError("dynamic mixing formula mismatch")
    if not dynamic.alpha.between(.05, .95).all() or not np.allclose(
            dynamic.loc[dynamic.as_of_date.eq(dynamic.as_of_date.min()), "alpha"], .3):
        raise AssertionError("dynamic alpha bounds or initial weight mismatch")
    dates = pd.DatetimeIndex(sorted(static.as_of_date.unique()))
    frames = {"Static": static.rename(columns={"pred_signal": "prediction_score"}),
              "Continual": continual.rename(columns={"pred_signal": "prediction_score"}),
              "Dynamic": dynamic.assign(prediction_score=dynamic.final_signal)}
    returns = _load_return_reference(static_dir / "predictions.csv")
    portfolio_cfg = PortfolioConfig(30, 3, 5, .0015)
    daily, summaries, curves = {}, [], {"as_of_date": dates}
    for name, source in frames.items():
        frame = source.drop(columns=["actual_return_1"], errors="ignore").merge(
            returns, on=["as_of_date", "stock"], validate="one_to_one")
        signals = daily_signal_metrics(frame, "prediction_score")
        _, top10 = portfolio_metrics(frame, "path", "prediction_score", "actual_signal", 20261013)
        top10["as_of_date"] = pd.to_datetime(top10.as_of_date)
        port, port_daily = _backtest(frame, name, portfolio_cfg)
        port_daily["as_of_date"] = pd.to_datetime(port_daily.as_of_date)
        combined = signals.merge(top10[["as_of_date", "top10_net"]], on="as_of_date", validate="one_to_one")
        combined = combined.merge(port_daily[["as_of_date", "net_return"]], on="as_of_date", validate="one_to_one")
        combined = combined.sort_values("as_of_date").reset_index(drop=True)
        if not pd.DatetimeIndex(combined.as_of_date).equals(dates) or not np.isfinite(
                combined[["rankic", "q5_q1", "top10_net", "net_return"]].to_numpy()).all():
            raise AssertionError(f"nonfinite or misaligned paired daily metrics: {name}")
        daily[name] = combined
        _atomic_csv(out / f"{name}_daily_metrics.csv", combined)
        _atomic_csv(out / f"{name}_top30_daily_returns.csv", port_daily)
        summaries.append({"method": name, "path_rankic": float(combined.rankic.mean()),
            "path_ic": float(combined.pearson_ic.mean()), "q5_q1": float(combined.q5_q1.mean()),
            "top10pct_path_net": float(combined.top10_net.mean()), **port})
        curves[name] = np.cumprod(1+combined.net_return.to_numpy()) - 1
        if abs(curves[name][-1] - port["net_cumulative_return"]) > 1e-10:
            raise AssertionError("cumulative curve differs from portfolio backtest")
    _atomic_csv(out / "comparison_metrics.csv", pd.DataFrame(summaries))
    index = paired_indices(len(dates))
    differences, evidence = {"as_of_date": dates}, []
    for name in ("Dynamic", "Continual"):
        for metric in ("rankic", "q5_q1", "top10_net", "net_return"):
            delta = daily[name][metric].to_numpy() - daily["Static"][metric].to_numpy()
            differences[f"{name}_minus_Static_{metric}"] = delta
            ci = np.quantile(delta[index].mean(axis=1), [.025, .975])
            evidence.append({"comparison": f"{name}-Static", "metric": metric,
                "mean_delta": float(delta.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                "newey_west_lag9_t": newey_west_tstat_mean(delta, 9)})
    for metric in ("rankic", "q5_q1", "top10_net", "net_return"):
        delta = daily['Dynamic'][metric].to_numpy() - daily['Continual'][metric].to_numpy()
        differences[f'Dynamic_minus_Continual_{metric}'] = delta
        ci = np.quantile(delta[index].mean(axis=1), [.025, .975])
        evidence.append({'comparison': 'Dynamic-Continual', 'metric': metric,
            'mean_delta': float(delta.mean()), 'ci_low': float(ci[0]), 'ci_high': float(ci[1]),
            'newey_west_lag9_t': newey_west_tstat_mean(delta, 9)})
    _atomic_csv(out / "paired_daily_differences.csv", pd.DataFrame(differences))
    _atomic_csv(out / "paired_block10_bootstrap.csv", pd.DataFrame(evidence))
    _atomic_csv(out / "cumulative_net_return.csv", pd.DataFrame(curves))
    # These are bootstrap statistics of realized returns, not rerun holdings paths.
    nonlinear = []
    def stats(values):
        wealth = np.cumprod(1+values, axis=-1)
        peaks = np.maximum(1., np.maximum.accumulate(wealth, axis=-1))
        std = values.std(axis=-1, ddof=1)
        return {"net_cumulative_return": wealth[..., -1]-1,
                "sharpe": np.divide(values.mean(axis=-1), std, out=np.zeros_like(std), where=std > 0)*np.sqrt(252),
                "maximum_drawdown": (wealth/peaks-1).min(axis=-1)}
    boot_s = stats(daily["Static"].net_return.to_numpy()[index])
    boot_d = stats(daily["Dynamic"].net_return.to_numpy()[index])
    point_s = stats(daily["Static"].net_return.to_numpy())
    point_d = stats(daily["Dynamic"].net_return.to_numpy())
    for metric in boot_s:
        ci = np.quantile(boot_d[metric]-boot_s[metric], [.025, .975])
        nonlinear.append({"comparison": "Dynamic-Static", "metric": metric,
            "delta": float(point_d[metric]-point_s[metric]), "ci_low": float(ci[0]), "ci_high": float(ci[1])})
    _atomic_csv(out / "top30_paired_bootstrap.csv", pd.DataFrame(nonlinear))
    if plt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
        colors = {"Static": "#365d85", "Dynamic": "#176b67", "Continual": "#a76518"}
        for name in ("Static", "Dynamic", "Continual"):
            axes[0].plot(dates, curves[name]*100, label=name, color=colors[name], linewidth=1.7,
                         alpha=.9 if name != "Continual" else .6)
            returns_array = daily[name].net_return.to_numpy()
            wealth = np.cumprod(1+returns_array)
            axes[1].plot(dates, (wealth/np.maximum(1, np.maximum.accumulate(wealth))-1)*100, color=colors[name])
        axes[0].set_title("D-RankIC: Top30/Drop3 net cumulative return (initial continual alpha = 0.3)")
        axes[0].set_ylabel("Net return (%)"); axes[0].legend()
        axes[1].set_ylabel("Drawdown (%)")
        weight_daily = dynamic[["as_of_date", "alpha"]].drop_duplicates().sort_values("as_of_date")
        axes[2].step(weight_daily.as_of_date, weight_daily.alpha, where="post", color="#176b67")
        axes[2].set_ylabel("Continual weight"); axes[2].set_ylim(0, 1)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(out / "cumulative_return_and_weights.png", dpi=170)
        fig.savefig(out / "cumulative_return_and_weights.svg")
        plt.close(fig)
    else:
        # Always deliver a viewable curve even in the minimal remote Python environment.
        colors = {"Static": "#365d85", "Dynamic": "#176b67", "Continual": "#a76518"}
        width, height, left, right = 1200, 760, 95, 30
        xs = np.linspace(left, width-right, len(dates))
        series = {name: curves[name]*100 for name in ("Static", "Dynamic", "Continual")}
        ymin = min(float(values.min()) for values in series.values())
        ymax = max(float(values.max()) for values in series.values())
        lo, hi = ymin-min(1., (ymax-ymin)*.08), ymax+max(1., (ymax-ymin)*.08)
        def poly(values, y0, h, vmin, vmax):
            return " ".join(f"{x:.1f},{y0+h-(v-vmin)/(vmax-vmin)*h:.1f}" for x,v in zip(xs, values))
        chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                  '<rect width="100%" height="100%" fill="white"/>',
                  '<text x="40" y="32" font-family="sans-serif" font-size="20">D-RankIC: Top30/Drop3 net cumulative return (initial continual alpha = 0.3)</text>']
        for tick in np.linspace(lo, hi, 5):
            y=60+350-(tick-lo)/(hi-lo)*350
            chunks.extend([f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>',
                           f'<text x="12" y="{y+4:.1f}" font-family="sans-serif" font-size="12">{tick:.1f}%</text>'])
        for name, values in series.items():
            chunks.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" points="{poly(values,60,350,lo,hi)}"/>')
        for j,(name,color) in enumerate(colors.items()):
            x=left+j*180
            chunks.extend([f'<line x1="{x}" y1="435" x2="{x+24}" y2="435" stroke="{color}" stroke-width="3"/>',
                           f'<text x="{x+30}" y="440" font-family="sans-serif" font-size="14">{name}</text>'])
        chunks.append('<text x="40" y="475" font-family="sans-serif" font-size="16">Drawdown (%)</text>')
        for name in ("Static", "Dynamic", "Continual"):
            wealth=np.cumprod(1+daily[name].net_return.to_numpy())
            dd=(wealth/np.maximum(1,np.maximum.accumulate(wealth))-1)*100
            chunks.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="1.5" points="{poly(dd,485,120,min(-25,float(dd.min())),0)}"/>')
        chunks.append('<text x="40" y="650" font-family="sans-serif" font-size="16">Continual weight</text>')
        alpha=dynamic.sort_values('as_of_date').groupby('as_of_date').alpha.first().to_numpy()
        chunks.append(f'<polyline fill="none" stroke="#176b67" stroke-width="2" points="{poly(alpha,660,75,0,1)}"/>')
        chunks.append('</svg>')
        (out/'cumulative_return_and_weights.svg').write_text("\n".join(chunks),encoding="utf-8")
    report = {"status": "complete", "evaluation": "retrospective_prequential_single_seed",
        "rows": 66103, "dates": 226, "seed": 100, "summaries": summaries, "paired_evidence": evidence,
        "bootstrap": {"block": 10, "repetitions": 5000, "seed": 20261013},
        "limitations": ["previously inspected fixed test period; one seed",
            "Rank-Gaussian mixed signal is dimensionless; no mixed MAE/RMSE",
            "Hedge learns from historical deployed prediction stream, not newly fitted retrospective predictions",
            "Top10% path returns overlap; only Top30 next-day series is plotted as cumulative net return",
            "nonlinear bootstrap resamples realized returns without rerunning holdings"]}
    report['evaluation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['top10_cost_rate'] = .001
    report['top30_cost_rate'] = .0015
    _atomic_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static-dir", type=Path, required=True)
    p.add_argument("--continual-dir", type=Path, required=True)
    p.add_argument("--dynamic-dir", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.static_dir, a.continual_dir, a.dynamic_dir)
