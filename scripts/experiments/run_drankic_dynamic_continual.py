"""Prequential D-RankIC continual learner with initial 0.3 Hedge signal weight."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from run_adapter_paired_purged_promotion_beta8 import (
    Model, _hashes, _key_set, _save_static, attach_label_interval,
    deploy_after_gate, enough_gate_days, fit_candidate_anchor_only,
    load_split, recent_holdout, select_purged_replay, seed_everything,
    verify_boundaries, verify_checkpoint_hash, verify_locked_keys, verify_purged,
)
from run_dynamic_continual_csi300 import (
    _load_and_join, _atomic_csv, _atomic_json, build_dynamic_predictions,
)
from run_adapter_continual_csi300_90x10 import (
    PRED_LEN, TEST_END, TEST_START, atomic_csv, atomic_json, metrics, predict,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--initial-checkpoint", type=Path, required=True)
    p.add_argument("--expected-initial-checkpoint-sha256", required=True)
    p.add_argument("--static-output-dir", type=Path, required=True)
    p.add_argument("--reference-static-dir", type=Path)
    p.add_argument("--continual-output-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--checkpoint-selection", default="D_rankic_epoch3")
    p.add_argument("--initial-training-summary", type=Path,
                   help="Offline selection_summary.json for the initial D checkpoint")
    p.add_argument("--hard-acceptance-gate", action="store_true",
                   help="Deploy only candidates passing the configured three-condition gate")
    p.add_argument("--recent-rankic-improvement", type=float, default=.002,
                   help="Minimum candidate minus deployed recent RankIC; negative values allow decline")
    p.add_argument("--anchor-rankic-tolerance", type=float, default=.005)
    p.add_argument("--recent-mae-tolerance", type=float, default=.05)
    return p.parse_args()


def _require_empty(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _run(cfg):
    thresholds = [cfg.recent_rankic_improvement, cfg.anchor_rankic_tolerance, cfg.recent_mae_tolerance]
    if not np.isfinite(thresholds).all() or min(thresholds[1:]) < 0:
        raise ValueError("gate thresholds must be finite; anchor and MAE tolerances must be nonnegative")
    deployment_policy = ("extreme_degradation_only" if cfg.recent_rankic_improvement < 0
                         else "accepted_only_gate") if cfg.hard_acceptance_gate else "direct_candidate_deployment_no_gate"
    for p in (cfg.static_output_dir, cfg.continual_output_dir, cfg.output_dir):
        _require_empty(p)
    seed_everything(cfg.seed)
    import run_adapter_paired_purged_promotion_beta8 as beta8
    beta_cfg = {
        "cache_dir": str(cfg.cache_dir), "panel_dir": str(cfg.panel_dir),
        "initial_checkpoint": str(cfg.initial_checkpoint),
        "expected_initial_checkpoint_sha256": cfg.expected_initial_checkpoint_sha256,
        "reference_checkpoint": str(cfg.initial_checkpoint),
        "checkpoint_selection": cfg.checkpoint_selection,
        "static_output_dir": str(cfg.static_output_dir),
        "continual_output_dir": str(cfg.continual_output_dir),
        "mode": "all", "device": cfg.device, "seed": cfg.seed,
        "batch_size": 16, "update_epochs": cfg.update_epochs, "head_lr": 3e-4,
        "adapter_lr": 1e-4, "weight_decay": 1e-4, "lambda_rank": .2,
        "patience": 2, "replay_max_rows": 120000,
        "replay_recent_months": 24, "replay_recent_fraction": .7,
        "recent_validation_days": 30, "minimum_recent_validation_days": 20,
        "recent_rankic_improvement": .002, "anchor_rankic_tolerance": .005,
        "recent_mae_tolerance": .05,
    }
    # Keep beta8's historical preregistration intact; this runner may explicitly
    # extend candidate training to ten epochs under a new experiment identity.
    if cfg.update_epochs not in (2, 10):
        raise ValueError("candidate epochs must be 2 (historical) or 10 (Page1-C protocol)")
    beta8.enforce_protocol(dict(beta_cfg, update_epochs=2, seed=100))
    # Gate overrides belong to this experiment, not beta8's historical protocol.
    beta_cfg.update(recent_rankic_improvement=cfg.recent_rankic_improvement,
                    anchor_rankic_tolerance=cfg.anchor_rankic_tolerance,
                    recent_mae_tolerance=cfg.recent_mae_tolerance,
                    deployment_policy=deployment_policy)
    hashes = _hashes(cfg.cache_dir, cfg.panel_dir, cfg.initial_checkpoint, Path(__file__))
    script_hashes = {str(p.relative_to(ROOT)): beta8.sha256_file(p) for p in (
        Path(__file__), Path(beta8.__file__),
        ROOT / 'scripts/experiments/run_adapter_continual_csi300_90x10.py',
        ROOT / 'scripts/experiments/run_dynamic_continual_csi300.py')}
    verify_checkpoint_hash(hashes["initial_checkpoint"], cfg.expected_initial_checkpoint_sha256)
    if cfg.initial_training_summary is not None:
        offline_selection = json.loads(cfg.initial_training_summary.read_text(encoding="utf-8"))
        if offline_selection["selection"]["rankic"]["checkpoint_sha256"] != hashes["initial_checkpoint"]:
            raise AssertionError("offline RankIC selection provenance does not match initial checkpoint")
        beta_cfg["initial_training"] = {k: offline_selection[k] for k in (
            "group", "seed", "epochs_max", "patience", "early_stop_metric", "selection")}
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    environment = {'python': platform.python_version(), 'pytorch': torch.__version__,
                   'cuda': torch.version.cuda, 'device': str(device),
                   'torch_num_threads': torch.get_num_threads(),
                   'KMP_DUPLICATE_LIB_OK': os.environ.get('KMP_DUPLICATE_LIB_OK'),
                   'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                   'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS')}
    for output in (cfg.static_output_dir, cfg.continual_output_dir, cfg.output_dir):
        atomic_json(output / 'environment.json', environment)
        atomic_json(output / 'script_hashes.json', script_hashes)
    train_h, train_y, train_m = load_split(cfg.cache_dir, "train")
    val_h, val_y, val_m = load_split(cfg.cache_dir, "val")
    test_h, test_y, test_m = load_split(cfg.cache_dir, "test")
    train_m, val_m, test_m = tuple(
        attach_label_interval(m, cfg.panel_dir) for m in (train_m, val_m, test_m)
    )
    for split, meta in (("train", train_m), ("val", val_m), ("test", test_m)):
        verify_locked_keys(meta, cfg.panel_dir, split, True)
    verify_boundaries(train_m, val_m, test_m, True)
    dates = pd.DatetimeIndex(sorted(test_m.as_of_date.unique()))
    if len(test_m) != 66103 or len(dates) != 226 or dates.min() != TEST_START or dates.max() != TEST_END:
        raise AssertionError("locked test panel differs from the registered 226-date / 66103-row window")
    state = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=True)
    model_static = Model(512, PRED_LEN, 64, .1).to(device)
    model_static.load_state_dict(state, strict=True)
    model_static.requires_grad_(False)
    static_pred = _save_static(model_static, test_h, test_y, test_m, device,
                               cfg.static_output_dir, beta_cfg, hashes)
    if cfg.reference_static_dir is not None:
        reference_config = json.loads((cfg.reference_static_dir / "run_config.json").read_text(encoding="utf-8"))
        for name in hashes:
            if name != "runner" and reference_config["hashes"].get(name) != hashes[name]:
                raise AssertionError(f"existing D-RankIC Static input identity mismatch: {name}")
        reference_pred = pd.read_csv(cfg.reference_static_dir / "predictions.csv", dtype={"stock": str})
    else:
        reference_pred = static_pred.copy()
    reference_pred = reference_pred.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    reproduced = static_pred.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    if not reference_pred[["as_of_date", "stock"]].equals(reproduced[["as_of_date", "stock"]]):
        raise AssertionError("existing D-RankIC Static keys differ")
    value_columns = (["pred_signal", "pred_endpoint_return", "actual_signal", "actual_endpoint_return"]
                     + [f"pred_return_{i}" for i in range(1, 11)]
                     + [f"actual_return_{i}" for i in range(1, 11)])
    if not np.allclose(reference_pred[value_columns], reproduced[value_columns], atol=1e-6, rtol=0):
        raise AssertionError("existing D-RankIC Static values were not reproduced within 1e-6")
    actual_columns = ['actual_signal', 'actual_endpoint_return'] + [f'actual_return_{i}' for i in range(1, 11)]
    if not np.array_equal(reference_pred[actual_columns].to_numpy(), reproduced[actual_columns].to_numpy()):
        # CSV float parsing can change the final bit; require tight, explicit label equality.
        if not np.allclose(reference_pred[actual_columns], reproduced[actual_columns], atol=1e-15, rtol=0):
            raise AssertionError('Static reference labels differ')
    # Freeze the exact existing baseline artifact; newly generated predictions served only as the audit.
    static_pred = reference_pred
    atomic_csv(cfg.static_output_dir / "predictions.csv", static_pred)
    model = Model(512, PRED_LEN, 64, .1).to(device)
    model.load_state_dict(state, strict=True)
    continual_initial = pd.DataFrame(predict(model, test_h, test_y, test_m, device, 0,
                                   False, "adapter_dpaired_continual", cfg.seed))
    pred_cols = [f"pred_return_{i}" for i in range(1, 11)]
    continual_initial = continual_initial.sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    if not static_pred[["as_of_date", "stock"]].astype(str).equals(
            continual_initial[["as_of_date", "stock"]].astype(str)):
        raise AssertionError("Static/Continual version_000 keys do not match")
    init_diff = float(np.max(np.abs(static_pred[pred_cols].to_numpy()
                                  - continual_initial[pred_cols].to_numpy())))
    if init_diff > 1e-6:
        raise AssertionError(f"initial paired prediction max diff {init_diff} exceeds 1e-6")
    cont_out = cfg.continual_output_dir
    for name in ("checkpoints", "replay_audit", "gate_audit"):
        (cont_out / name).mkdir()
    torch.save(state, cont_out / "checkpoints" / "version_000.pt")
    atomic_json(cont_out / "initial_pair_audit.json", {
        "status": "passed", "rows": len(test_m), "dates": len(dates),
        "max_abs_diff": init_diff, "initial_checkpoint_sha256": hashes["initial_checkpoint"]})
    print('[INITIAL_PAIR]', json.dumps({'max_abs_diff': init_diff, 'rows': len(test_m), 'dates': len(dates)}), flush=True)
    month_ends = {dates[i] for i in range(len(dates)-1) if dates[i].month != dates[i+1].month}
    seen: set[int] = set()
    rows, logs, version = [], [], 0
    anchor = beta8.validation_metrics(model, val_h, val_y, val_m, device)
    start_time = time.time()
    for pos, date in enumerate(dates):
        dm = test_m[test_m.as_of_date.eq(date)].copy()
        rows.extend(predict(model, test_h, test_y, dm, device, version, False,
                            "adapter_drankic_dynamic_continual", cfg.seed))
        if date not in month_ends:
            continue
        gate, old_mature, stats = recent_holdout(test_m, date, 30)
        available = set(test_m[(test_m.as_of_date < date) &
                               (test_m.label_end_date <= date)]._row_id.astype(int))
        new_rows = len(available - seen)
        seen.update(available)
        log = {"update_date": date.date().isoformat(),
               "next_test_date": dates[pos+1].date().isoformat(),
               "model_version_before": version, **stats,
               "new_matured_test_rows": new_rows, "candidate_trained": False,
               "promotion_ran": False,
               "recent_rankic_threshold": beta_cfg["recent_rankic_improvement"],
               "anchor_rankic_tolerance": beta_cfg["anchor_rankic_tolerance"],
               "recent_mae_tolerance": beta_cfg["recent_mae_tolerance"]}
        if not enough_gate_days(stats, beta_cfg):
            logs.append({**log, "decision": "skipped_insufficient_recent_validation",
                         "model_version_after": version, "selected_replay_rows": 0})
            continue
        chosen, replay_stats = select_purged_replay(
            train_m, old_mature, gate, date, beta_cfg, cfg.seed + pos)
        train_part, test_part = chosen[chosen._source.eq("train")], chosen[chosen._source.eq("test")]
        ti, xi = train_part._row_id.to_numpy(dtype=np.int64), test_part._row_id.to_numpy(dtype=np.int64)
        uh = np.concatenate([np.asarray(train_h[ti]), np.asarray(test_h[xi])])
        uy = np.concatenate([np.asarray(train_y[ti]), np.asarray(test_y[xi])])
        um = pd.concat([train_part, test_part], ignore_index=True)
        if _key_set(um) & _key_set(gate) or _key_set(um) & _key_set(val_m):
            raise AssertionError("gate or anchor validation row leaked into candidate replay")
        verify_purged(um, gate)
        current_gate = beta8.validation_metrics(model, test_h, test_y, gate, device)
        current_anchor = beta8.validation_metrics(model, val_h, val_y, val_m, device)
        atomic_csv(cont_out / "replay_audit" / f"{date.date()}.csv",
                   chosen[["as_of_date", "stock", "label_start_date", "label_end_date", "_source", "_row_id"]])
        atomic_csv(cont_out / "gate_audit" / f"{date.date()}.csv",
                   gate[["as_of_date", "stock", "label_start_date", "label_end_date", "_row_id"]])
        candidate = copy.deepcopy(model).to(device)
        seed_everything(cfg.seed + pos + 1)
        candidate, history, _, best_epoch = beta8.fit_candidate_anchor_only(
            candidate, uh, uy, um, val_h, val_y, val_m, beta_cfg, device, cfg.seed + pos + 1)
        candidate_gate = beta8.validation_metrics(candidate, test_h, test_y, gate, device)
        candidate_anchor = beta8.validation_metrics(candidate, val_h, val_y, val_m, device)
        gate_ok = True
        gate_detail = {}
        if cfg.hard_acceptance_gate:
            gate_ok, gate_detail = beta8.gate_decision(
                {"recent": current_gate, "anchor": current_anchor},
                {"recent": candidate_gate, "anchor": candidate_anchor},
                beta_cfg,
            )
        promoted, promotion = deploy_after_gate(
            gate_ok, model, candidate, chosen, gate, train_h, train_y, test_h, test_y,
            beta_cfg, device, cfg.seed + pos + 10000, date)
        if gate_ok:
            version += 1
            torch.save(candidate.state_dict(), cont_out / "checkpoints" / f"candidate_gate_{date.date()}.pt")
            torch.save(promoted.state_dict(), cont_out / "checkpoints" / f"version_{version:03d}.pt")
            promotion.update({
                "deployed_checkpoint": f"checkpoints/version_{version:03d}.pt",
                "candidate_gate_checkpoint": f"checkpoints/candidate_gate_{date.date()}.pt",
                "deployment_effective_date": dates[pos+1].date().isoformat(),
            })
            model = promoted
            anchor = beta8.validation_metrics(model, val_h, val_y, val_m, device)
        logs.append({**log, **replay_stats, "candidate_trained": True,
            "decision": ("accepted_then_promoted" if cfg.hard_acceptance_gate else "trained_promoted_no_hard_gate") if gate_ok else "rejected_validation_gate",
            "accepted": gate_ok, "model_version_after": version,
            "best_epoch": best_epoch, "candidate_gate_rankic": candidate_gate["rankic"],
            "current_gate_rankic": current_gate["rankic"],
            "current_gate_standardized_mae": current_gate["standardized_mae"],
            "candidate_gate_standardized_mae": candidate_gate["standardized_mae"],
            "current_anchor_rankic": current_anchor["rankic"],
            "candidate_anchor_rankic": candidate_anchor["rankic"],
            "anchor_after_promotion_rankic": anchor["rankic"],
            "candidate_gate_rows": len(gate),
            "purge_boundary_label_start_date": gate.label_start_date.min().date().isoformat(),
            "purge_train_max_label_end_date": chosen.label_end_date.max().date().isoformat(),
            "selected_replay_rows": len(chosen),
            "selected_max_label_end_date": chosen.label_end_date.max().date().isoformat(),
            "history": history, **gate_detail, **promotion})
        print("[UPDATE]", json.dumps({"date": log["update_date"], "version": version,
              "best_epoch": best_epoch, "gate_rankic": candidate_gate["rankic"],
              "accepted": gate_ok}), flush=True)
        atomic_csv(cont_out / 'update_log.csv', pd.DataFrame(logs))
        atomic_json(cont_out / 'progress.json', {'last_update_date': log['update_date'],
            'model_version': version, 'elapsed_seconds': time.time() - start_time})
    continual_pred = pd.DataFrame(rows).sort_values(["as_of_date", "stock"]).reset_index(drop=True)
    if len(continual_pred) != 66103 or continual_pred[["as_of_date", "stock"]].duplicated().any():
        raise AssertionError("continual predictions have missing or duplicate keys")
    if not np.isfinite(continual_pred[["pred_signal"] + pred_cols].to_numpy()).all():
        raise AssertionError("continual predictions contain nonfinite values")
    atomic_csv(cont_out / "predictions.csv", continual_pred)
    atomic_csv(cont_out / "daily_metrics.csv", metrics(continual_pred))
    atomic_csv(cont_out / "update_log.csv", pd.DataFrame(logs))
    atomic_json(cont_out / "run_config.json", {
        **beta_cfg, "branch": "D_rankic_hard_gate_continual" if cfg.hard_acceptance_gate else "D_rankic_no_hard_gate_continual",
        "initial_checkpoint_sha256": hashes["initial_checkpoint"], "hashes": hashes,
        "device_resolved": str(device), "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()), "test_dates": len(dates), "test_rows": len(continual_pred),
        "hard_acceptance_gate": bool(cfg.hard_acceptance_gate), "gate_rows_excluded_from_candidate": True,
        "historical_checkpoint_sha256_available": False, "current_checkpoint_authorized_by_user": True,
        "historical_identity_check": "current_authorized_checkpoint_hash_and_existing_static_predictions_verified",
        "deployment_policy": deployment_policy,
        "script_hashes": script_hashes,
        "promotion": "after candidate gate acceptance" if cfg.hard_acceptance_gate else "after successful candidate training",
        "promotion_epochs": 1, "promotion_head_lr": 3e-5, "promotion_adapter_lr": 1e-5,
        "candidate_epoch_selection": "highest fixed anchor RankIC; patience=2"})
    atomic_json(cont_out / "summary.json", {"status": "complete", "updates": len(logs),
        "promotions": sum(bool(x.get("promotion_ran")) for x in logs),
        "final_version": version, "elapsed_seconds": time.time() - start_time})
    joined = _load_and_join(cfg.static_output_dir / "predictions.csv", cont_out / "predictions.csv")
    mixed, weights = build_dynamic_predictions(joined, test_start=str(TEST_START.date()),
        test_end=str(TEST_END.date()), history_days=60, eta=.5, alpha_min=.05,
        alpha_max=.95, rank_weight=.7, error_weight=.3, initial_alpha=.3)
    mixed["prediction_score"] = mixed["final_signal"]
    _atomic_csv(cfg.output_dir / "predictions.csv", mixed)
    _atomic_csv(cfg.output_dir / "weight_log.csv", weights)
    blend = {"initial_continual_weight": .3, "initial_static_weight": .7,
        "history_days": 60, "eta": .5, "rank_weight": .7, "error_weight": .3,
        "alpha_min": .05, "alpha_max": .95, "mean_alpha": float(mixed.alpha.mean()),
        "min_alpha": float(mixed.alpha.min()), "max_alpha": float(mixed.alpha.max())}
    _atomic_json(cfg.output_dir / "run_config.json", {"status": "complete",
        "seed": cfg.seed, "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()),
        "dates": len(dates), "rows": len(mixed), "checkpoint_sha256": hashes["initial_checkpoint"],
        "hashes": hashes, "hard_acceptance_gate": bool(cfg.hard_acceptance_gate), "blend": blend,
        "deployment_policy": deployment_policy,
        "gate": {key: beta_cfg[key] for key in ("recent_rankic_improvement", "anchor_rankic_tolerance", "recent_mae_tolerance")},
        "script_hashes": script_hashes, "historical_checkpoint_sha256_available": False,
        "current_checkpoint_authorized_by_user": True,
        "weight_update": "month_end_after_prediction; mature labels only; effective next test date"})
    _atomic_json(cfg.output_dir / "summary.json", {"status": "complete", **blend,
        "promotions": sum(bool(x.get("promotion_ran")) for x in logs), "update_count": len(logs)})
    return mixed, weights


if __name__ == "__main__":
    _run(parse_args())
