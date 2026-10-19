from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.metrics import EvaluationConfig, clean_json, evaluate_frame
from evaluate.inputs import normalize_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_predictions(source: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"no prediction CSV files in {source}")
    frames = [pd.read_csv(path, dtype={"stock": str, "stock_id": str, "symbol": str, "code": str}) for path in files]
    return normalize_predictions(pd.concat(frames, ignore_index=True)), files


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _write_result(raw: pd.DataFrame, sources: list[Path], output: Path, config: EvaluationConfig,
                  *, aer_report: Path | None = None, update_log: Path | None = None) -> dict:
    result = evaluate_frame(raw, config)
    result["sources"] = [{"file": str(source.resolve()), "sha256": _sha256(source)} for source in sources]
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["evaluator_sources"] = [{"file": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                                   for path in sorted(Path(__file__).parent.glob("*.py"))]
    if aer_report:
        report = pd.read_csv(aer_report)
        required = {"as_of_date", "return", "benchmark", "cost"}
        if not required.issubset(report):
            raise ValueError(f"AER report needs columns {sorted(required)}")
        report["as_of_date"] = pd.to_datetime(report.as_of_date, errors="raise").dt.strftime("%Y-%m-%d")
        if report.as_of_date.duplicated().any():
            raise ValueError("AER report has duplicate dates")
        if config.test_start:
            report = report[report.as_of_date >= config.test_start]
        if config.test_end:
            report = report[report.as_of_date <= config.test_end]
        if set(report.as_of_date) != set(raw.loc[
            raw.as_of_date.between(config.test_start or "0000", config.test_end or "9999"), "as_of_date"]):
            raise ValueError("AER report dates differ from prediction test dates")
        values = report[["return", "benchmark", "cost"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("AER report has missing/non-finite returns, benchmark or cost")
        result["metrics"]["kronos_aer"] = {
            "status": "可验证（外部逐日报告复算）", "aer": float((values[:, 0] - values[:, 1] - values[:, 2]).mean() * 238),
            "annualization_days": 238, "dates": len(values),
            "execution_verified": False, "benchmark_definition": "caller-supplied benchmark; verify against experiment configuration",
            "source": str(aer_report.resolve()), "sha256": _sha256(aer_report),
            "note": "仅复算外部报告的年化公式，不认证报告的成交、成本或基准生成过程",
        }
        report.to_csv(output / "aer_report.csv", index=False)
    if update_log:
        updates = pd.read_csv(update_log)
        updates.to_csv(output / "validation_gate_log.csv", index=False)
        result["metrics"]["validation_rankic_gate"] = {
            "status": "日志已归档；未认证训练边界或接纳决策", "rows": len(updates),
            "source": str(update_log.resolve()), "sha256": _sha256(update_log), "is_test_metric": False,
        }

    base = "evaluation"
    details = result.pop("daily", {})
    metrics = result["metrics"]
    for mode, tables in details.items():
        cross_daily = pd.DataFrame(tables["cross_section"])
        if cross_daily.empty:
            cross_daily = pd.DataFrame(columns=["as_of_date", "n", "ic", "rankic"])
        portfolio_daily = pd.DataFrame(tables["portfolio"])
        if portfolio_daily.empty:
            portfolio_daily = pd.DataFrame(columns=["as_of_date", "n", "top_raw", "top_net", "turnover"])
        cross_daily.to_csv(output / f"{base}_{mode}_cross_section_daily.csv", index=False)
        portfolio_daily.to_csv(output / f"{base}_{mode}_portfolio_daily.csv", index=False)
    pd.DataFrame(metrics["trajectory_by_horizon"]).to_csv(output / f"{base}_point_by_horizon.csv", index=False)
    _write(output / f"{base}_summary.json", json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    summary_rows = []
    for mode in ("path", "endpoint"):
        section = metrics.get(mode, {})
        for family in ("point", "direction", "cross_section", "portfolio"):
            entry = section.get(family, {})
            summary_rows.append({"mode": mode, "family": family, **_flatten(entry)})
    summary_rows.append({"mode": "trajectory_pooled", "family": "point", **metrics["trajectory_pooled"]})
    summary_rows.append({"mode": "independent_aer", "family": "portfolio", **_flatten(metrics["kronos_aer"])})
    summary_rows.append({"mode": "validation", "family": "model_selection", **_flatten(metrics["validation_rankic_gate"])})
    pd.DataFrame(summary_rows).to_csv(output / f"{base}_summary.csv", index=False)
    return {"sources": [str(source) for source in sources], "summary": str(output / f"{base}_summary.json"),
            "rows": result["config"]["observed_rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="计算股票预测的 Path/Endpoint 指标并生成可复现摘要。")
    parser.add_argument("--predictions", nargs="+", required=True, type=Path,
                        help="预测 CSV，或仅含分日期预测 CSV 的目录；多源输入分别评测")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="CSI300", choices=("CSI300", "CSI800", "S&P500", "Yahoo S&P 500"))
    parser.add_argument("--test-start", default="2025-07-01")
    parser.add_argument("--test-end", default="2026-06-05")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-fraction", type=float)
    parser.add_argument("--drop-n", type=int)
    parser.add_argument("--min-holding-days", type=int, default=5)
    parser.add_argument("--cost-rate", type=float)
    parser.add_argument("--annualization-days", type=int, help="CSI800 默认238，其他市场默认252")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-block", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--group", help="仅评测输入文件 group 列中的指定实验组，例如 C")
    parser.add_argument("--return-reference", type=Path, help="提供下一日真实收益的预测 CSV；按日期/股票匹配，仅补充 actual_return_1")
    parser.add_argument("--reference-keys", type=Path, help="固定测试面板 CSV（as_of_date,stock）；只评测这些键，缺少任何键时报错")
    parser.add_argument("--aer-report", type=Path, help="外部已执行回测的日频报告：as_of_date,return,benchmark,cost")
    parser.add_argument("--update-log", type=Path, help="归档已有验证门/更新 CSV 日志")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的非空输出目录")
    args = parser.parse_args()

    sources = [path.resolve() for path in args.predictions]
    if len(set(sources)) != len(sources):
        parser.error("预测文件不能重复")
    missing = [path for path in sources if not path.exists()]
    if missing:
        parser.error(f"预测文件不存在: {missing}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"输出目录非空；为避免覆盖，请指定新目录，或显式传入 --overwrite: {output}")
    config = EvaluationConfig(
        dataset=args.dataset, test_start=args.test_start, test_end=args.test_end,
        top_k=args.top_k, top_k_fraction=args.top_k_fraction, drop_n=args.drop_n,
        min_holding_days=args.min_holding_days, cost_rate=args.cost_rate,
        annualization_days=args.annualization_days, bootstrap_reps=args.bootstrap_reps,
        bootstrap_block=args.bootstrap_block, seed=args.seed,
    )
    prepared = []
    key_reference = read_predictions(args.reference_keys)[0] if args.reference_keys else None
    if key_reference is not None:
        key_reference["as_of_date"] = pd.to_datetime(key_reference.as_of_date, errors="raise").dt.strftime("%Y-%m-%d")
        if config.test_start:
            key_reference = key_reference[key_reference.as_of_date >= config.test_start]
        if config.test_end:
            key_reference = key_reference[key_reference.as_of_date <= config.test_end]
        key_reference = key_reference[["as_of_date", "stock"]]
        if key_reference.empty or key_reference.duplicated(["as_of_date", "stock"]).any():
            parser.error("固定测试面板必须含非空、唯一的日期/股票键")
    reference = read_predictions(args.return_reference)[0] if args.return_reference else None
    if reference is not None:
        reference["as_of_date"] = pd.to_datetime(reference.as_of_date, errors="raise").dt.strftime("%Y-%m-%d")
        if reference.duplicated(["as_of_date", "stock"]).any() or "actual_return_1" not in reference:
            parser.error("收益参考必须有唯一日期/股票键和 actual_return_1 列")
    for source in sources:
        frame, files = read_predictions(source)
        frame["as_of_date"] = pd.to_datetime(frame.as_of_date, errors="raise").dt.strftime("%Y-%m-%d")
        if args.group:
            if "group" not in frame:
                parser.error(f"--group requires a group column: {source}")
            frame = frame[frame.group.astype(str).eq(args.group)].copy()
        split_cols = [column for column in ("group", "seed", "method", "model", "experiment_type")
                      if column in frame and frame[column].nunique(dropna=False) > 1]
        if split_cols:
            parser.error(f"输入包含多项实验或种子 {split_cols}；请拆分 CSV 或用 --group 选择实验组: {source}")
        if reference is not None:
            column = "actual_return_1"
            joined = frame.merge(reference[["as_of_date", "stock", column]], on=["as_of_date", "stock"],
                                 how="left", validate="one_to_one", suffixes=("", "_reference"))
            if f"{column}_reference" in joined:
                left, right = pd.to_numeric(joined[column], errors="coerce"), pd.to_numeric(joined[f"{column}_reference"], errors="coerce")
                both = np.isfinite(left) & np.isfinite(right)
                if not np.allclose(left[both], right[both], atol=1e-7, rtol=0):
                    parser.error("收益参考与输入中已有的 actual_return_1 冲突")
                joined[column] = left.where(np.isfinite(left), right)
                joined = joined.drop(columns=[f"{column}_reference"])
            frame = joined
        input_rows = len(frame)
        if key_reference is not None:
            frame = frame.merge(key_reference, on=["as_of_date", "stock"], how="inner", validate="one_to_one")
            if len(frame) != len(key_reference):
                parser.error(f"预测缺少固定测试面板中的 {len(key_reference)-len(frame)} 个键: {source}")
        prepared.append((frame, files, input_rows))
    if (args.aer_report or args.update_log) and len(prepared) != 1:
        parser.error("--aer-report / --update-log 仅支持单实验输入，以免错配外部报告")
    labels = [f"{source.parent.name}_{source.stem}" for source in sources]
    if len(set(labels)) != len(labels):
        parser.error("批量输入目录/文件名重复，请分开运行")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for label, (frame, files, input_rows) in zip(labels, prepared):
        destination = output / label if len(prepared) > 1 else output
        destination.mkdir(parents=True, exist_ok=True)
        item = _write_result(frame, files, destination, config, aer_report=args.aer_report, update_log=args.update_log)
        if args.reference_keys:
            reference_files = read_predictions(args.reference_keys)[1]
            item["alignment"] = {"input_rows": input_rows, "aligned_rows": len(frame),
                                 "reference_sources": [{"file": str(p.resolve()), "sha256": _sha256(p)} for p in reference_files]}
            summary_path = Path(item["summary"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["alignment"] = item["alignment"]
            _write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        if args.return_reference:
            item["return_reference"] = {"file": str(args.return_reference.resolve()),
                                        "sources": [{"file": str(p), "sha256": _sha256(p)} for p in read_predictions(args.return_reference)[1]]}
            summary_path = Path(item["summary"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["return_reference"] = item["return_reference"]
            _write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        results.append(item)
    _write(output / "evaluation_index.json", json.dumps(clean_json({"config": config.__dict__, "results": results}),
                                                        ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"output_dir": str(output), "results": [
        {"summary": item["summary"], "rows": item["rows"], "source_file_count": len(item["sources"])}
        for item in results
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
