"""Causal S/F/A/D pilot; saved forecasts are evaluated by the public evaluate CLI.

The training population deliberately retains the original prefix-pilot selection.
The prediction/evaluation population is the full locked CSI300 panel.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional

from scripts.experiments import run_delayed_feedback_pilot as old
from src.doubleadapt.compare import indices, return_stats

KEYS = ["as_of_date", "stock"]
PCOLS = [f"pred_return_{k}" for k in range(1, 11)]
ACOLS = [f"actual_return_{k}" for k in range(1, 11)]
GROUPS = ("S", "F", "A", "D")


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = old.Adapter(4, 2)
        self.head = nn.Linear(4, 3)

    def forward(self, h):
        return self.head(self.adapter(h))


class ReleasedLabels:
    """Only scalar labels are exposed, after both maturity checks succeed."""

    def __init__(self, values, releases):
        self.values, self.releases = values, releases

    def read(self, row_id, k, today):
        dates = self.releases[row_id]
        if dates[k] > today or dates[-1] > today:
            raise AssertionError("unreleased complete-path label access")
        value = float(self.values[row_id, k])
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite mature label")
        return value


def configure(model, group, head_lr=1e-4, adapter_lr=1e-5):
    model.eval()  # Keep dropout disabled even for the trainable adapter.
    for p in model.parameters():
        p.requires_grad_(False)
    parameter_groups = []
    if group != "S":
        for p in model.head.parameters():
            p.requires_grad_(True)
        parameter_groups.append({"params": list(model.head.parameters()), "lr": head_lr})
    # Zero adapter LR explicitly freezes it: its unused gradients must not
    # change the global clipping of the head in the A -> F identity check.
    if group == "A" and adapter_lr > 0:
        for p in model.adapter.parameters():
            p.requires_grad_(True)
        parameter_groups.append({"params": list(model.adapter.parameters()), "lr": adapter_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0) if parameter_groups else None
    return optimizer, [p for p in model.parameters() if p.requires_grad]


def normalized_state(z):
    unit = functional.normalize(z, dim=1)
    mean = unit.mean(0)
    norm = torch.linalg.vector_norm(mean)
    if not torch.isfinite(mean).all() or not torch.isfinite(norm) or norm <= 1e-12:
        return None
    return mean / norm


def sample_weights(z, state, uniform=False):
    if uniform or state is None:
        return torch.ones(len(z)), torch.zeros(len(z)), "forced_uniform" if uniform else "zero_state"
    similarity = functional.normalize(z, dim=1) @ state
    a = torch.exp(torch.clamp((similarity - similarity.mean()) / .1, -math.log(2), math.log(2)))
    weights = a / a.mean()
    if not torch.isfinite(weights).all() or weights.min() < .25 - 1e-6 or weights.max() > 4 + 1e-6:
        raise AssertionError("invalid normalized similarity weights")
    return weights, similarity, "similarity"


def make_schedule(dates, training_rows, horizon=10, max_rows=32):
    """Exactly retain the previous F's per-horizon F/P stock intersection."""
    schedule = {}
    for pos, today in enumerate(dates):
        events = []
        if pos >= horizon:
            complete = [d for d in dates[:pos] if len(training_rows[d]) and training_rows[d].iloc[0].label_dates[-1] == today]
            for k in range(horizon):
                prefix = [d for d in dates[:pos] if len(training_rows[d]) and training_rows[d].iloc[0].label_dates[k] == today]
                if not complete or not prefix:
                    continue
                origin, prefix_origin = complete[-1], prefix[-1]
                frows, prows = training_rows[origin], training_rows[prefix_origin]
                stocks = sorted(set(frows.stock) & set(prows.stock), key=old.stable_stock_key)[:max_rows]
                for stock in stocks:
                    row = frows.loc[frows.stock.eq(stock)].iloc[0]
                    if not (origin < today and row.label_dates[-1] <= today):
                        raise AssertionError("invalid training event metadata")
                    events.append({"row_id": int(row.row_id), "stock": stock, "origin_date": origin,
                                   "horizon": k + 1, "label_date": row.label_dates[k],
                                   "complete_date": row.label_dates[-1]})
        schedule[today] = events
    return schedule


def run_stream(group, model, h_by_date, z_by_date, rows, schedule, labels, out,
               *, head_lr=1e-4, adapter_lr=1e-5, uniform=False, stop_after=None, resume=None):
    """Prediction shards are committed before any supervised access on that day.

    Compact daily checkpoints reference per-day journals in the same run directory.
    Resume truncates the logical stream at the requested checkpoint and deterministically
    rewrites only later shards. The CLI never resumes/overwrites an existing experiment.
    """
    out = Path(out)
    optimizer, trainable = configure(model, group, head_lr, adapter_lr)
    live_adapter = group == "A" and adapter_lr > 0
    dates, consumed, start, version = sorted(rows), set(), 0, 0
    pred_parts, access_parts, update_parts, weight_parts = [], [], [], []
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["group"] != group:
            raise AssertionError("resume group mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        consumed = set(map(tuple, state["consumed"]))
        start, version = state["next_pos"], state["version"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        for d in dates[:start]:
            pred_parts.append(pd.read_csv(out / "prediction_shards" / group / f"{d}.csv", float_precision="round_trip"))
            for kind, parts in (("access", access_parts), ("update", update_parts), ("weight", weight_parts)):
                file = out / "journals" / group / f"{d}_{kind}.csv"
                if file.exists():
                    parts.append(pd.read_csv(file, float_precision="round_trip"))
    for pos, today in enumerate(dates[start:], start):
        model.eval()
        with torch.no_grad():
            prediction = (model(h_by_date[today]) if live_adapter else model.head(z_by_date[today])).numpy()
        frame = rows[today][KEYS + ["row_id"]].copy()
        frame["group"], frame["pred_version"] = group, version
        for k in range(prediction.shape[1]):
            frame[f"pred_return_{k + 1}"] = prediction[:, k].astype(np.float64)
        shard = out / "prediction_shards" / group / f"{today}.csv"
        old.atomic_csv(shard, frame)
        prediction_hash = old.sha256(shard)
        pred_parts.append(frame)
        events = schedule[today] if group != "S" else []
        accesses, weights_log = [], []
        if events:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            adapter_grad_norm = 0.0
            state = normalized_state(z_by_date[today]) if group == "D" else None
            for k in sorted({e["horizon"] for e in events}):
                selected = [e for e in events if e["horizon"] == k]
                ids = []
                targets = []
                # Metadata is fixed before accessing any supervised scalar.
                for event in selected:
                    key = (event["row_id"], k)
                    if key in consumed:
                        raise AssertionError("duplicate label consumption")
                    targets.append(labels.read(event["row_id"], k - 1, today))
                    consumed.add(key)
                    ids.append(event["row_id"])
                    accesses.append({"group": group, "update_date": today, **event,
                                     "prediction_sha256": prediction_hash})
                origin = selected[0]["origin_date"]
                lookup = {int(r): i for i, r in enumerate(rows[origin].row_id)}
                selected_pos = [lookup[i] for i in ids]
                z = z_by_date[origin][selected_pos]
                output = (model(h_by_date[origin][selected_pos]) if live_adapter else model.head(z))[:, k - 1]
                element_loss = functional.smooth_l1_loss(output, torch.tensor(targets), beta=1, reduction="none")
                if group == "D":
                    w, similarity, reason = sample_weights(z, state, uniform)
                    for event, value, sim in zip(selected, w.tolist(), similarity.tolist()):
                        weights_log.append({"update_date": today, **event, "weight": value,
                                            "similarity": sim, "reason": reason})
                else:
                    w = torch.ones(len(selected))
                losses.append((w * element_loss).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            for p in trainable:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise FloatingPointError("missing/nonfinite gradient")
            if live_adapter:
                adapter_grad_norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.adapter.parameters()))
            grad_norm = float(nn.utils.clip_grad_norm_(trainable, 1))
            optimizer.step()
            version += 1
            access = pd.DataFrame(accesses)
            update = pd.DataFrame([{"group": group, "date": today, "supervision": len(events),
                "optimizer_steps": 1, "unique_feature_rows": access.row_id.nunique(),
                "loss": float(loss.detach()), "grad_norm": grad_norm, "adapter_grad_norm": adapter_grad_norm,
                "version": version, "consumed_events": len(consumed)}])
            for kind, data, parts in (("access", access, access_parts), ("update", update, update_parts),
                                      ("weight", pd.DataFrame(weights_log), weight_parts)):
                if not data.empty:
                    old.atomic_csv(out / "journals" / group / f"{today}_{kind}.csv", data)
                    parts.append(data)
            checkpoint = out / "checkpoints" / group / f"{today}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"group": group, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "consumed": sorted(consumed), "next_pos": pos + 1, "version": version,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()}, checkpoint)
        if stop_after is not None and pos + 1 >= stop_after:
            break
    result = [pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
              for parts in (pred_parts, access_parts, update_parts, weight_parts)]
    for suffix, frame in zip(("predictions", "access_log", "update_log", "weight_log"), result):
        if not frame.empty:
            old.atomic_csv(out / group / f"{suffix}.csv", frame)
    return result


def correctness_checks():
    """Small fixtures call the same engine as the real experiment."""
    old.seed_all(100)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dates = [f"2020-01-{i + 1:02d}" for i in range(20)]
    run_dates = dates[:14]
    initial = ToyModel()
    h = {d: torch.randn(5, 4) for d in run_dates}
    initial.eval()
    with torch.no_grad():
        z = {d: initial.adapter(x) for d, x in h.items()}
    rows = {d: pd.DataFrame([{"as_of_date": d, "stock": f"s{i}", "row_id": j * 5 + i,
                             "label_dates": dates[j + 1:j + 4]} for i in range(5)]) for j, d in enumerate(run_dates)}
    releases = {int(r.row_id): r.label_dates for frame in rows.values() for r in frame.itertuples()}
    values = np.random.default_rng(100).normal(0, .01, (70, 3))
    schedule = make_schedule(run_dates, rows, 3, 4)
    checks = []
    with tempfile.TemporaryDirectory(prefix="module_pilot_") as task_dir:
        root = Path(task_dir)
        def run(name, group, data=values, table=rows, hh=h, zz=z, **kwargs):
            model = copy.deepcopy(initial)
            result = run_stream(group, model, hh, zz, table, schedule,
                                ReleasedLabels(data, releases), root / name, **kwargs)
            return model, result
        models, records = {}, {}
        for group in GROUPS:
            models[group], records[group] = run(group, group)
        for group in ("F", "A", "D"):
            access, updates = records[group][1:3]
            assert len(updates) == 11 and len(access) == 132
            assert (access.complete_date <= access.update_date).all()
            assert not access.duplicated(["row_id", "horizon"]).any()
            for d, digest in access[["update_date", "prediction_sha256"]].drop_duplicates().itertuples(index=False, name=None):
                assert old.sha256(root / group / "prediction_shards" / group / f"{d}.csv") == digest
            pd.testing.assert_frame_equal(access.drop(columns=["group", "prediction_sha256"]),
                                          records["F"][1].drop(columns=["group", "prediction_sha256"]))
        checks.append("maturity_single_consumption_matched_events_steps_prediction_commit")
        assert not old.nested_equal(models["A"].adapter.state_dict(), initial.adapter.state_dict())
        assert (records["A"][2].adapter_grad_norm > 0).all()
        assert old.nested_equal(models["D"].adapter.state_dict(), initial.adapter.state_dict())
        checks.append("A_adapter_gradients_and_updates_D_adapter_frozen")
        for group, kwargs in (("A", {"adapter_lr": 0}), ("D", {"uniform": True})):
            model, rec = run(group + "_identity", group, **kwargs)
            assert old.nested_equal(model.state_dict(), models["F"].state_dict())
            np.testing.assert_array_equal(rec[0].filter(regex="^pred_return_"), records["F"][0].filter(regex="^pred_return_"))
        checks.append("zero_adapter_lr_A_and_uniform_D_equal_F")
        for group in ("F", "A", "D"):
            model, rec = run(group + "_zero", group, head_lr=0, adapter_lr=0)
            np.testing.assert_array_equal(rec[0].filter(regex="^pred_return_"), records["S"][0].filter(regex="^pred_return_"))
        checks.append("zero_lr_equals_static")
        cutoff = run_dates[7]
        poison = values.copy()
        for row_id, dd in releases.items():
            for k, release in enumerate(dd):
                if release > cutoff:
                    poison[row_id, k] = np.nan
        for group in ("F", "A", "D"):
            clean, cr = run(group + "_cut", group, stop_after=8)
            bad, br = run(group + "_poison", group, data=poison, stop_after=8)
            assert old.nested_equal(clean.state_dict(), bad.state_dict())
            for a, b in zip(cr, br):
                pd.testing.assert_frame_equal(a, b)
            cp = root / (group + "_cut") / "checkpoints" / group / f"{cutoff}.pt"
            bp = root / (group + "_poison") / "checkpoints" / group / f"{cutoff}.pt"
            assert old.nested_equal(torch.load(cp, weights_only=False)["optimizer"], torch.load(bp, weights_only=False)["optimizer"])
            resumed, rr = run(group + "_cut", group, resume=cp)
            assert old.nested_equal(resumed.state_dict(), models[group].state_dict())
            for a, b in zip(rr, records[group]):
                pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-14, rtol=1e-14)
            final = torch.load(root / group / "checkpoints" / group / f"{run_dates[-1]}.pt", weights_only=False)
            again = torch.load(root / (group + "_cut") / "checkpoints" / group / f"{run_dates[-1]}.pt", weights_only=False)
            assert old.nested_equal(final["optimizer"], again["optimizer"])
            assert final["consumed"] == again["consumed"]
        checks.append("future_poison_no_effect_and_resume_identical_all_online_groups")
        # Permute source rows with their features; canonical schedule must preserve training.
        for group in ("F", "A", "D"):
            model, rec = run(group + "_reversed", group,
                table={d: f.iloc[::-1].reset_index(drop=True) for d, f in rows.items()},
                hh={d: x.flip(0) for d, x in h.items()}, zz={d: x.flip(0) for d, x in z.items()})
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, models[group].state_dict()[key], atol=1e-7, rtol=0)
            np.testing.assert_allclose(rec[0].sort_values(KEYS).filter(regex="^pred_return_"),
                records[group][0].sort_values(KEYS).filter(regex="^pred_return_"), atol=1e-7, rtol=0)
        checks.append("same_day_input_order_invariance")
        for zz, state in ((torch.zeros(5, 4), None), (torch.randn(5, 4), torch.zeros(4))):
            w, _, _ = sample_weights(zz, state)
            torch.testing.assert_close(w, torch.ones(5))
        weights = records["D"][3]
        assert weights.weight.between(.25 - 1e-6, 4 + 1e-6).all()
        np.testing.assert_allclose(weights.groupby(["update_date", "horizon"]).weight.mean(), 1, atol=1e-6)
        checks.append("D_weight_bounds_mean_one_and_zero_state_fallback")
    return {"status": "passed", "checks": checks, "engine": "actual run_stream", "fixture_horizon": 3}


def main(args):
    started = time.perf_counter()
    checks = correctness_checks()
    print(json.dumps(checks), flush=True)
    if args.checks_only:
        return
    old.seed_all(100)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("new output directory required; old runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    old.atomic_json(out / "correctness_report.json", checks)
    if old.sha256(args.checkpoint) != old.EXPECTED_CKPT:
        raise AssertionError("initial checkpoint hash mismatch")
    meta, test, hidden, labels, calendar, positions, local, label_dates = old.load_inputs(
        args.cache_dir, args.reference_dir, args.hidden_dir)
    dates = calendar[:60]
    assert dates[0] == "2025-07-01" and dates[-1] == "2025-09-22"
    stocks, training_panel = old.select_panel(meta, test, calendar)
    panel = test[test.as_of_date.isin(dates)].copy()
    reference_path = ROOT / "experiments/csi300_aligned/models/adapter_drankic_dynamic_static_20261013/predictions.csv"
    reference = pd.read_csv(reference_path, dtype={"stock": str})
    reference = reference[reference.as_of_date.isin(dates)].sort_values(KEYS).reset_index(drop=True)
    if not reference[KEYS].equals(panel.sort_values(KEYS).reset_index(drop=True)[KEYS]):
        raise AssertionError("full locked reference panel mismatch; no intersections allowed")
    initial = old.make_model(args.checkpoint, False)
    hidden_hash = old.sha256(args.hidden_dir / "hidden_test.npy")
    historical_config = json.loads((reference_path.parent / "run_config.json").read_text())
    if hidden_hash != historical_config["hashes"]["hidden_test.npy"]:
        raise AssertionError("hidden cache mismatch")
    z_all = old.features(initial, hidden, np.arange(len(test)))
    rows, h_by_date, z_by_date, training_rows = {}, {}, {}, {}
    for d in dates:
        part = panel[panel.as_of_date.eq(d)].sort_values("stock").copy().reset_index(drop=True)
        part["label_dates"] = [label_dates[d]] * len(part)
        if not part.label_end_date.eq(label_dates[d][-1]).all():
            raise AssertionError("label maturity metadata mismatch")
        rows[d] = part
        ix = np.array([local[int(r)] for r in part.row_id])
        h_by_date[d] = torch.tensor(np.asarray(hidden[ix]), dtype=torch.float32)
        z_by_date[d] = z_all[ix]
        training_rows[d] = part[part.stock.isin(stocks)].reset_index(drop=True)
    schedule = make_schedule(dates, training_rows)
    releases = {int(r.row_id): r.label_dates for frame in rows.values() for r in frame.itertuples()}
    old.atomic_csv(out / "reference_keys.csv", reference[KEYS])
    old.atomic_csv(out / "training_schedule.csv", pd.DataFrame(
        [{"update_date": d, **event} for d, events in schedule.items() for event in events]))
    old.atomic_json(out / "run_config.json", {"groups": GROUPS, "checkpoint_sha256": old.EXPECTED_CKPT,
        "hidden_sha256": hidden_hash, "reference_sha256": old.sha256(reference_path), "dates": dates,
        "score_dates": dates[10:], "training_stocks": stocks, "max_labels_per_horizon": 32,
        "head_lr": 1e-4, "adapter_lr": 1e-5, "weight_decay": 0, "gradient_clip": 1,
        "smooth_l1_beta": 1, "dropout": "disabled", "seed": 100, "temperature": .1,
        "similarity_log_clip": [-math.log(2), math.log(2)], "weight_bounds": [.25, 4],
        "primary_evaluation": "50 scoring dates; start from cash on score date 1; entry cost included",
        "secondary_evaluation": "all 60 dates; separate cash reset",
        "half_windows": "slice the primary realized portfolio without rebuilding holdings",
        "endpoint_guard": "endpoint cumulative net return and mean IC both >= F",
        "half_guard": "each half: Path cumulative net return > F OR Path mean IC > F",
        "retrospective": True, "compute_matched": False, "no_tuning": True,
        "source_sha256": old.sha256(Path(__file__)), "original_engine_sha256": old.sha256(Path(old.__file__))})
    shutil.copy2(__file__, out / "executed_source.py")
    old.atomic_json(out / "status.json", {"status": "running"})
    group_records, group_stats = {}, []
    for group in GROUPS:
        t = time.perf_counter()
        model = copy.deepcopy(initial)
        result = run_stream(group, model, h_by_date, z_by_date, rows, schedule,
                            ReleasedLabels(labels, releases), out)
        group_records[group] = result
        head_delta = math.sqrt(sum(float(((p.detach() - q.detach()) ** 2).sum()) for p, q in zip(model.head.parameters(), initial.head.parameters())))
        adapter_delta = math.sqrt(sum(float(((p.detach() - q.detach()) ** 2).sum()) for p, q in zip(model.adapter.parameters(), initial.adapter.parameters())))
        if group != "A" and adapter_delta != 0:
            raise AssertionError("frozen adapter changed")
        if group == "A" and (adapter_delta <= 0 or not (result[2].adapter_grad_norm > 0).all()):
            raise AssertionError("A adapter not learning")
        group_stats.append({"group": group, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "head_parameter_delta_l2": head_delta, "adapter_parameter_delta_l2": adapter_delta,
            "updates": len(result[2]), "supervision": len(result[1]), "seconds": time.perf_counter() - t})
        print(json.dumps({"stage": "group_complete", **group_stats[-1]}), flush=True)
    old.atomic_csv(out / "training_summary.csv", pd.DataFrame(group_stats))
    audit = audit_run(out, group_records, reference, training_panel, dates, labels)
    old.atomic_json(out / "real_run_audit.json", audit)
    # Full labels are attached only AFTER every group's causal forecast stream has finished.
    for group, (prediction, *_) in group_records.items():
        evaluation_frame = prediction.merge(reference[KEYS + ACOLS], on=KEYS, validate="one_to_one")
        evaluation_frame["pred_signal"] = evaluation_frame[PCOLS].mean(axis=1)
        evaluation_frame["actual_signal"] = evaluation_frame[ACOLS].mean(axis=1)
        for prefix in ("pred", "actual"):
            evaluation_frame[f"{prefix}_path_mean_return"] = evaluation_frame[f"{prefix}_signal"]
            evaluation_frame[f"{prefix}_endpoint_return"] = evaluation_frame[f"{prefix}_return_10"]
        old.atomic_csv(out / group / "predictions.csv", evaluation_frame)
    evaluate_and_report(out, dates, group_stats)
    old.atomic_json(out / "status.json", {"status": "complete", "seconds": time.perf_counter() - started,
        "prediction_rows_per_group": len(panel), "score_rows_per_group": int(panel.as_of_date.isin(dates[10:]).sum()),
        "score_dates": 50, "groups": GROUPS})
    files = [p for p in out.rglob("*") if p.is_file() and "checkpoints" not in p.parts]
    old.atomic_json(out / "artifact_manifest.json", {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": old.sha256(p)} for p in files})
    print(f"COMPLETE {out}", flush=True)


def audit_run(out, records, reference, training_panel, dates, labels):
    errors = {}
    for group in ("F", "A", "D"):
        access, updates = records[group][1:3]
        assert len(updates) == 50 and len(access) == 16000
        assert (access.label_date <= access.update_date).all()
        assert (access.complete_date <= access.update_date).all()
        assert not access.duplicated(["row_id", "horizon"]).any()
        pd.testing.assert_frame_equal(access.drop(columns=["group", "prediction_sha256"]),
                                      records["F"][1].drop(columns=["group", "prediction_sha256"]))
        pd.testing.assert_frame_equal(updates[["date", "supervision", "optimizer_steps"]],
                                      records["F"][2][["date", "supervision", "optimizer_steps"]])
        for d, digest in access[["update_date", "prediction_sha256"]].drop_duplicates().itertuples(index=False, name=None):
            assert old.sha256(out / "prediction_shards" / group / f"{d}.csv") == digest
    for group in GROUPS:
        prediction = records[group][0].sort_values(KEYS).reset_index(drop=True)
        assert prediction[KEYS].equals(reference[KEYS])
        assert not prediction.duplicated(KEYS).any()
        assert np.isfinite(prediction[PCOLS].to_numpy()).all()
        if group == "S":
            errors["historical_static_max_abs_error"] = float(np.max(np.abs(prediction[PCOLS].to_numpy() - reference[PCOLS].to_numpy())))
            assert errors["historical_static_max_abs_error"] <= 1e-7
        # Same training cohort as previous pilot; only evaluation population changed.
        if group in ("S", "F"):
            previous = pd.read_csv(ROOT / f"experiments/csi300_aligned/models/delayed_feedback_pilot_20261020/{group}_predictions.csv")
            paired = previous.merge(prediction, on=KEYS, suffixes=("_old", "_new"), validate="one_to_one")
            assert len(paired) == len(previous)
            error = max(float(np.max(np.abs(paired[f"{c}_old"] - paired[f"{c}_new"]))) for c in PCOLS)
            assert error <= 1e-7, f"previous {group} mismatch: {error}"
            errors[f"previous_{group}_max_abs_error"] = error
    # Verify targets across the whole prediction panel, not just training stocks.
    ids = reference.merge(records["S"][0][KEYS + ["row_id"]], on=KEYS, validate="one_to_one")
    assert len(ids) == len(reference)
    errors["historical_label_max_abs_error"] = float(np.max(np.abs(ids[ACOLS].to_numpy() - np.asarray(labels[ids.row_id.to_numpy()]))))
    assert errors["historical_label_max_abs_error"] <= 1e-7
    weights = records["D"][3]
    daily_weights = weights.groupby(["update_date", "horizon"]).weight.agg(["min", "max", "mean", "std"])
    assert daily_weights["min"].min() >= .25 - 1e-6 and daily_weights["max"].max() <= 4 + 1e-6
    np.testing.assert_allclose(daily_weights["mean"], 1, atol=1e-6)
    effective = weights.groupby(["update_date", "horizon"]).weight.apply(lambda w: w.sum() ** 2 / np.square(w).sum())
    old.atomic_csv(out / "D_weight_diagnostics.csv", daily_weights.join(effective.rename("effective_sample_size")).reset_index())
    old.atomic_json(out / "coverage_audit.json", {"rows_per_group": len(reference), "score_rows": int(reference.as_of_date.isin(dates[10:]).sum()),
        "prediction_stocks": int(reference.stock.nunique()), "training_stocks_observed": int(training_panel.stock.nunique()),
        "counts_by_date": reference.groupby("as_of_date").size().to_dict(), "same_keys_all_groups": True})
    return {"status": "passed", **errors, "matched_events_and_steps": True, "future_reads": 0,
            "duplicate_consumption": 0, "prediction_hashes_verified": True,
            "D_weight_min": float(weights.weight.min()), "D_weight_max": float(weights.weight.max()),
            "D_mean_effective_sample_size": float(effective.mean())}


def evaluate_and_report(out, dates, group_stats, *, run_evaluator=True):
    """Use the public CLI unchanged; comparison code consumes its daily artifacts."""
    for window, start in (("primary_50", dates[10]), ("full_60", dates[0])):
        if not run_evaluator:
            continue
        command = [sys.executable, "-m", "evaluate", "--predictions", *[str(out / g / "predictions.csv") for g in GROUPS],
            "--dataset", "CSI300", "--output-dir", str(out / "evaluation" / window),
            "--test-start", start, "--test-end", dates[-1], "--reference-keys", str(out / "reference_keys.csv"),
            "--top-k", "30", "--drop-n", "3", "--min-holding-days", "5", "--cost-rate", ".0015",
            "--annualization-days", "252", "--bootstrap-reps", "5000", "--bootstrap-block", "10", "--seed", "100"]
        subprocess.run(command, cwd=ROOT, check=True)
    daily, summary, half_rows = {}, [], []
    for group in GROUPS:
        for mode in ("path", "endpoint"):
            directory = out / "evaluation" / "primary_50" / f"{group}_predictions"
            cross = pd.read_csv(directory / f"evaluation_{mode}_cross_section_daily.csv")
            portfolio = pd.read_csv(directory / f"evaluation_{mode}_portfolio_daily.csv")
            frame = cross.merge(portfolio, on="as_of_date", validate="one_to_one", suffixes=("", "_portfolio")).sort_values("as_of_date").reset_index(drop=True)
            assert frame.as_of_date.tolist() == dates[10:]
            assert np.isfinite(frame[["ic", "rankic", "top_net", "turnover"]].to_numpy()).all()
            np.testing.assert_allclose(frame.top_net, frame.top_raw - .0015 * frame.turnover, atol=1e-14, rtol=0)
            assert frame.turnover.iloc[0] == 1
            daily[group, mode] = frame
            row = {"group": group, "signal": mode, "ic": frame.ic.mean(), "rankic": frame.rankic.mean(),
                   "turnover": frame.turnover.mean(), **{k: float(v) for k, v in return_stats(frame.top_net.to_numpy()).items()}}
            summary.append(row)
            for name, sl in (("first_25", slice(0, 25)), ("last_25", slice(25, 50))):
                half = frame.iloc[sl]
                half_rows.append({"group": group, "signal": mode, "half": name,
                    "ic": half.ic.mean(), "rankic": half.rankic.mean(), **{k: float(v) for k, v in return_stats(half.top_net.to_numpy()).items()}})
    summary, halves = pd.DataFrame(summary), pd.DataFrame(half_rows)
    old.atomic_csv(out / "comparison_metrics.csv", summary)
    old.atomic_csv(out / "half_window_metrics.csv", halves)
    draws = indices(50, 5000, 10, 100)
    paired, bootstrap = [], []
    for left, right in (("F", "S"), ("A", "F"), ("D", "F"), ("A", "D")):
        for mode in ("path", "endpoint"):
            a, b = daily[left, mode], daily[right, mode]
            row = pd.DataFrame({"comparison": f"{left}_minus_{right}", "signal": mode, "as_of_date": a.as_of_date})
            for metric in ("ic", "rankic", "top_net", "turnover", "transaction_cost", "top_raw"):
                delta = a[metric].to_numpy() - b[metric].to_numpy()
                row[metric + "_delta"] = delta
                boot = delta[draws].mean(1)
                bootstrap.append({"comparison": f"{left}_minus_{right}", "signal": mode, "metric": "mean_" + metric,
                    "delta": float(delta.mean()), "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
            row["cumulative_net_return_delta"] = np.cumprod(1 + a.top_net) - np.cumprod(1 + b.top_net)
            paired.append(row)
            # Reuse the project's paired realized-return bootstrap; this does not
            # retrain models or rerun holdings on synthetic bootstrap calendars.
            bs_a, bs_b = return_stats(a.top_net.to_numpy()[draws]), return_stats(b.top_net.to_numpy()[draws])
            stat_a, stat_b = return_stats(a.top_net.to_numpy()), return_stats(b.top_net.to_numpy())
            for metric in stat_a:
                delta = bs_a[metric] - bs_b[metric]
                bootstrap.append({"comparison": f"{left}_minus_{right}", "signal": mode, "metric": metric,
                    "delta": float(stat_a[metric] - stat_b[metric]), "ci_low": float(np.quantile(delta, .025)), "ci_high": float(np.quantile(delta, .975))})
    old.atomic_csv(out / "paired_daily_differences.csv", pd.concat(paired, ignore_index=True))
    old.atomic_csv(out / "paired_block10_bootstrap.csv", pd.DataFrame(bootstrap))
    decisions = {}
    indexed = summary.set_index(["group", "signal"])
    half_index = halves.set_index(["group", "signal", "half"])
    for group in ("A", "D"):
        p, f = indexed.loc[(group, "path")], indexed.loc[("F", "path")]
        e, ef = indexed.loc[(group, "endpoint")], indexed.loc[("F", "endpoint")]
        half_checks = []
        for half in ("first_25", "last_25"):
            h, hf = half_index.loc[(group, "path", half)], half_index.loc[("F", "path", half)]
            half_checks.append(bool(h.net_cumulative_return > hf.net_cumulative_return or h.ic > hf.ic))
        checks = {"path_net_return_improves": bool(p.net_cumulative_return > f.net_cumulative_return),
            "path_ic_not_worse": bool(p.ic >= f.ic), "endpoint_return_and_ic_not_worse": bool(e.net_cumulative_return >= ef.net_cumulative_return and e.ic >= ef.ic),
            "drawdown_deterioration_at_most_5pp": bool(p.max_drawdown >= f.max_drawdown - .05),
            "both_half_windows_show_return_or_ic_improvement": all(half_checks)}
        conflict = (p.net_cumulative_return > f.net_cumulative_return and p.ic < f.ic) or (p.ic > f.ic and p.net_cumulative_return <= f.net_cumulative_return)
        decision = "positive_candidate" if all(checks.values()) else ("partial_metric_conflict" if conflict else "no_consistent_positive_optimization")
        decisions[group] = {"decision": decision, "checks": checks,
            "delta_path_net_return": float(p.net_cumulative_return - f.net_cumulative_return), "delta_path_ic": float(p.ic - f.ic)}
    old.atomic_json(out / "decision.json", {"candidates": decisions, "no_automatic_expansion": True,
        "boundary": "retrospective small-window screening; not stable profit or novelty evidence"})
    plot_results(out, daily)
    lines = ["# 完整CSI300面板：在线模块小实验", "",
        f"预测区间 {dates[0]}—{dates[-1]}；主要评分 {dates[10]}—{dates[-1]}，50日。完整60日为独立次要结果，不混入主表。",
        "组合在评分首日从现金建仓，初始换手1，计入15bps；前10日只作共同暖启动。前后25日指标切分主组合的已实现收益，不重新建仓。",
        "", "S=静态；F=完整成熟标签仅更新预测头；A=更新原Adapter和头；D=同F加当前特征相似度权重。",
        "全部使用同一main D-RankIC checkpoint。完整锁定面板预测，在线训练仍为原预选股票内每horizon最多32只。",
        "", "|模型|Path累计净收益|Path IC|Endpoint累计净收益|Endpoint IC|Path Sharpe|Path最大回撤|Path换手|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        p, e = indexed.loc[(group, "path")], indexed.loc[(group, "endpoint")]
        lines.append(f"|{group}|{p.net_cumulative_return:.3%}|{p.ic:.6f}|{e.net_cumulative_return:.3%}|{e.ic:.6f}|{p.sharpe:.3f}|{p.max_drawdown:.3%}|{p.turnover:.3%}|")
    lines += ["", "主指标来自原evaluate CLI；累计净收益由该CLI输出的top_net复利，IC为逐日Pearson均值。Path/Endpoint分别组成独立持仓，均只用actual_return_1兑现日收益。",
        "Top30/Drop3、最低持有5日、单边15bps，等权close-to-close代理；不模拟涨跌停、停牌成交或开盘执行。", "",
        "|比较|Path收益差（百分点）|Path IC差|Endpoint收益差（百分点）|Endpoint IC差|",
        "|---|---:|---:|---:|---:|"]
    for a, b in (("F", "S"), ("A", "F"), ("D", "F"), ("A", "D")):
        p, f = indexed.loc[(a, "path")], indexed.loc[(b, "path")]
        e, ef = indexed.loc[(a, "endpoint")], indexed.loc[(b, "endpoint")]
        lines.append(f"|{a}−{b}|{100 * (p.net_cumulative_return - f.net_cumulative_return):+.3f}|{p.ic - f.ic:+.6f}|{100 * (e.net_cumulative_return - ef.net_cumulative_return):+.3f}|{e.ic - ef.ic:+.6f}|")
    lines += ["", *[f"- {g} vs F：{d['decision']}；检查结果 {d['checks']}" for g, d in decisions.items()], "",
        "配对95%区间（仅作探索）：", "",
        "|比较|信号|指标|差值|95%下限|95%上限|", "|---|---|---|---:|---:|---:|"]
    for r in bootstrap:
        if r["comparison"] in ("A_minus_F", "D_minus_F") and r["metric"] in ("net_cumulative_return", "mean_ic"):
            scale = 100 if r["metric"] == "net_cumulative_return" else 1
            label = "净收益差（百分点）" if scale == 100 else "IC差"
            lines.append(f"|{r['comparison']}|{r['signal']}|{label}|{scale*r['delta']:+.6f}|{scale*r['ci_low']:+.6f}|{scale*r['ci_high']:+.6f}|")
    lines += ["", "本窗各组平均换手均为11%，按相同单边费率计费。D相对F的收益下降不能用更高的总换手成本解释；差异来自所持股票的已实现毛收益路径。具体日期见paired_daily_differences.csv。",
        "A虽然通过筛选，但其Path最大回撤较F增大、Sharpe降低；Endpoint累计收益仍低于S。它不是所有指标上最优。", "",
        "A解冻原共享Adapter是结构消融，D是特征相似度加权假说；本轮不认定其论文原创性，不叠加、不扫参、不自动扩展窗口。",
        "IC/净收益冲突时报告部分改善；即使正向筛选通过，也不是稳定收益结论。累计收益差以百分点报告，不对负收益做相对百分比提升。",
        "", "配对区块bootstrap：50日、block=10、5000次、seed=100；相同日期块比较IC与已实现日收益。收益区间未重训练模型或重建抽样路径的持仓，短窗统计不确定性较大；未作多重检验后的论文显著性宣称。",
        "本区间及旧固定成分股面板已多次查看，属于回顾性探索，保留幸存者/成分股前视偏差。", "",
        "![累计净收益](cumulative_net_returns.png)", "", "![IC](ic_curves.png)", "",
        "详细文件：comparison_metrics.csv、half_window_metrics.csv、paired_daily_differences.csv、paired_block10_bootstrap.csv、training_summary.csv、D_weight_diagnostics.csv、real_run_audit.json。",
        "完整评测指标和每日文件位于evaluation/primary_50及evaluation/full_60，包含点位、方向、横截面和组合指标。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(out, daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind, filename in (("returns", "cumulative_net_returns"), ("ic", "ic_curves")):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for ax, mode in zip(axes, ("path", "endpoint")):
            for group in GROUPS:
                frame = daily[group, mode]
                values = (np.cumprod(1 + frame.top_net) - 1) * 100 if kind == "returns" else frame.ic.rolling(5, min_periods=1).mean()
                line, = ax.plot(pd.to_datetime(frame.as_of_date), values, label=group)
                if kind == "ic":
                    ax.plot(pd.to_datetime(frame.as_of_date), frame.ic, color=line.get_color(), alpha=.15, lw=.7)
            ax.axhline(0, color="gray", lw=.7)
            ax.set_ylabel(f"{mode}: " + ("net return (%)" if kind == "returns" else "IC (5-day mean)"))
            ax.legend(ncol=4)
            ax.grid(alpha=.2)
        fig.suptitle("CSI300 full locked panel | 50 scored days | retrospective")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(out / f"{filename}.{extension}", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "experiments/csi300_aligned/doubleadapt_cache_v2")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "experiments/csi300_aligned/reference")
    parser.add_argument("--hidden-dir", type=Path, default=ROOT / "experiments/csi300_aligned/reference/hidden_cache")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".runtime/D_model_rankic_pilot_original.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/csi300_aligned/models/online_module_pilot_20261020")
    parser.add_argument("--checks-only", action="store_true")
    main(parser.parse_args())
