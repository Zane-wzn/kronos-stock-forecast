"""Atomic prequential blocks, accepted-only state commits, exact RNG restoration."""
from __future__ import annotations
import argparse
import io
import json
import os
import platform
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import (ROOT, KEYS, PCOLS, ACOLS, atomic_json, atomic_csv, sha256, preflight,
                   base_predict, build_states, history_pool, mature_partition)
from .page1c_mars_cl import MARS, finite, rng_state, restore_rng, state_hash

PROTOCOL = {"version": 1, "predict_days": 20, "update_days": 40, "recent_update_dates": 30,
            "shadow_dates": 10, "history_cap": 60000, "history_seed": 20261013,
            "loss_weights": {"path": .3, "rank": .5, "anchor": .2}, "pair_cap_per_date": 4096,
            "batch_unit": "one_complete_sampled_date", "epochs": 2, "lr": 1e-4,
            "weight_decay": 1e-4, "clip_grad": 1., "soft_write": .2, "residual_bound": .02,
            "rankic_tolerance": .002, "mse_ratio": 1.05,
            "accepted_optimizer_policy": "candidate Adam moments retained, parameters interpolated",
            "state": "raw 6 fields mean/std/last; train-only z-score without clipping",
            "historical_page1c_net_return": .6248, "retrospective": True}


def save_checkpoint(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({**payload, "rng": rng_state()}, tmp)
    # Serialize and restore before committing, including optimizer and location.
    restored = torch.load(tmp, map_location="cpu", weights_only=False)
    if state_hash(restored["engine"]) != state_hash(payload["engine"]):
        raise AssertionError("checkpoint roundtrip failed")
    os.replace(tmp, path)


def record_prediction(path, frame):
    """Existing shards are immutable, including an interrupted uncommitted block."""
    if path.exists():
        prior = pd.read_csv(path, dtype={"stock": str})
        if not prior[["row_id"]+KEYS].equals(frame[["row_id"]+KEYS].reset_index(drop=True)):
            raise AssertionError("prediction shard keys changed after interruption")
        if not np.allclose(prior[PCOLS], frame[PCOLS], rtol=0, atol=1e-6):
            raise AssertionError("historical predictions changed after interruption")
    else:
        atomic_csv(path, frame)


def label_frame(recorded, y, seed, method):
    ids = recorded.row_id.to_numpy(int)
    out = recorded.copy()
    actual = np.asarray(y[ids])
    finite(actual, "export labels")
    for k,c in enumerate(ACOLS):
        out[c] = actual[:, k]
    # Use the recorded predictions, including any pre-crash shards.
    out["pred_signal"] = out[PCOLS].to_numpy().mean(1)
    out["pred_endpoint_return"] = out[PCOLS[-1]]
    out["actual_signal"] = actual.mean(1)
    out["actual_endpoint_return"] = actual[:, -1]
    out["seed"], out["method"] = seed, method
    return out


def stream(engine, meta, base, states, y, history, out, seed, fingerprint, *, resume=False, stop_after_blocks=None):
    """The only accesses to test labels before export are selected mature row IDs."""
    out = Path(out)
    shards = out/"prediction_days"
    shards.mkdir(parents=True, exist_ok=True)
    latest = out/"checkpoints/online_latest.pt"
    logs, shadows, training, next_block = [], [], [], 0
    if resume:
        if not latest.exists():
            raise FileNotFoundError("--resume requires an initialized online checkpoint")
        saved = torch.load(latest, map_location=engine.device, weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise AssertionError("resume inputs/config/code differ")
        engine.load_state_dict(saved["engine"])
        restore_rng(saved["rng"])
        next_block, logs, shadows, training = saved["next_block"], saved["logs"], saved["shadows"], saved["training"]
    else:
        save_checkpoint(latest, engine=engine.state_dict(), next_block=0, logs=[], shadows=[], training=[], fingerprint=fingerprint, shard_hashes={})
    if resume:
        for name, expected_hash in saved["shard_hashes"].items():
            if sha256(shards/name) != expected_hash:
                raise AssertionError("committed prediction shard modified")
    test = meta[meta.split.eq("test")]
    dates = sorted(test.as_of_date.unique())
    # Only historical mature rows are read before online prediction starts.
    if (meta.loc[history, "label_end_date"] > "2024-12-15").any():
        raise AssertionError("immature history")
    for block_index, start in enumerate(range(0, len(dates), 20)):
        if block_index < next_block:
            continue
        block_dates = dates[start:start+20]
        for date in block_dates:
            part = test[test.as_of_date.eq(date)]
            ids = part.row_id.to_numpy(int)
            pred = engine.predict(base[ids], states[ids])
            frame = part[["row_id"]+KEYS+["label_start_date", "label_end_date"]].reset_index(drop=True).copy()
            frame[PCOLS] = pred
            record_prediction(shards/f"{date}.csv", frame)
        # Current block's predictions exist before ANY selected label read.
        if (start+len(block_dates)) % 40 == 0 and len(block_dates) == 20:
            date = block_dates[-1]
            update, shadow = mature_partition(test, date)
            row = {"update_date": date, "block": block_index, "prediction_committed_before_label_read": True,
                   "update_rows": len(update), "shadow_rows": len(shadow),
                   "update_dates": update.as_of_date.nunique(), "shadow_dates": shadow.as_of_date.nunique()}
            if shadow.as_of_date.nunique() < 10 or update.empty:
                row.update(decision="skipped_insufficient_complete_mature_dates", accepted=False)
            else:
                train_ids = np.concatenate([history, update.row_id.to_numpy(int)])
                val_ids = shadow.row_id.to_numpy(int)
                used = meta.iloc[np.concatenate([train_ids, val_ids])]
                if used.as_of_date.ge(date).any() or used.label_end_date.gt(date).any() or set(train_ids)&set(val_ids):
                    raise AssertionError("immature or overlapping update/shadow samples")
                membership = pd.concat([meta.iloc[history].assign(role="history"), update.assign(role="recent_update"), shadow.assign(role="shadow")])
                membership["update_date"] = date
                atomic_csv(out/"task_samples"/f"{date}.csv", membership)
                # No future/unselected label values are passed to the optimizer or gate.
                proposal, daily, hist = engine.propose(base[train_ids], states[train_ids], np.asarray(y[train_ids]),
                    meta.iloc[train_ids].as_of_date.to_numpy(), base[val_ids], states[val_ids], np.asarray(y[val_ids]),
                    meta.iloc[val_ids].as_of_date.to_numpy())
                row.update(proposal)
                row.update(max_update_label_end=meta.iloc[train_ids].label_end_date.max(),
                           max_shadow_label_end=shadow.label_end_date.max(), mature_rows_used=len(used))
                shadows += [{**v, "update_date": date} for v in daily]
                training += [{**v, "update_date": date} for v in hist]
            logs.append(row)
        hashes = {p.name: sha256(p) for p in sorted(shards.glob("*.csv"))}
        payload = dict(engine=engine.state_dict(), next_block=block_index+1, logs=logs, shadows=shadows,
                       training=training, fingerprint=fingerprint, shard_hashes=hashes)
        save_checkpoint(out/"checkpoints"/f"block_{block_index:03d}.pt", **payload)
        save_checkpoint(latest, **payload)
        atomic_csv(out/"update_log.csv", pd.DataFrame(logs))
        atomic_csv(out/"shadow_validation.csv", pd.DataFrame(shadows))
        atomic_csv(out/"training_history.csv", pd.DataFrame(training))
        atomic_json(out/"status.json", {"status": "running", "next_block": block_index+1, "seed": seed})
        print(json.dumps({"stage": "block", "block": block_index, "end": block_dates[-1], "updates": len(logs)}), flush=True)
        if stop_after_blocks is not None and block_index+1 >= stop_after_blocks:
            return None
    recorded = pd.concat([pd.read_csv(p, dtype={"stock": str}) for p in sorted(shards.glob("*.csv"))], ignore_index=True)
    if not recorded[KEYS].equals(test[KEYS].reset_index(drop=True)) or recorded.duplicated(KEYS).any():
        raise AssertionError("online prediction coverage differs from complete test")
    mars = label_frame(recorded, y, seed, "Page1C_MARS_CL")
    frozen = recorded.copy()
    frozen[PCOLS] = base[frozen.row_id.to_numpy(int)]
    frozen = label_frame(frozen, y, seed, "Page1C_Static")
    atomic_csv(out/"mars_cl/predictions.csv", mars)
    atomic_csv(out/"frozen/predictions.csv", frozen)
    # Complete row/update membership, including all immature and unused rows.
    queues = []
    for log in logs:
        date = log["update_date"]
        u, v = mature_partition(test, date)
        q = test[["row_id"]+KEYS+["label_end_date"]].copy()
        q["update_date"] = date
        q["role"] = np.where(q.as_of_date.lt(date)&q.label_end_date.le(date), "mature_unused", "not_mature_or_not_occurred")
        q.loc[q.row_id.isin(u.row_id), "role"] = "recent_update"
        q.loc[q.row_id.isin(v.row_id), "role"] = "shadow"
        queues.append(q)
    atomic_csv(out/"maturity_queue.csv", pd.concat(queues, ignore_index=True) if queues else pd.DataFrame())
    atomic_json(out/"status.json", {"status": "predictions_complete", "seed": seed, "rows": len(mars),
                "dates": mars.as_of_date.nunique(), "updates": len(logs), "accepted_updates": sum(bool(r["accepted"]) for r in logs)})
    return frozen, mars


def run(args):
    out = args.output_dir.resolve()
    if not args.dry_run and not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError("formal output must be new; use --resume for a committed run")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    audit, meta, frozen_model = preflight(args)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/"preflight.json", audit)
    atomic_json(out/"initial_prediction_identity.json", audit["identity"])
    if args.dry_run:
        atomic_json(out/"status.json", {"status": "dry_run_"+audit["status"], "blockers": audit["blockers"]})
        print(json.dumps({"status": audit["status"], "identity": audit["identity"], "blockers": audit["blockers"]}, indent=2))
        return
    if audit["blockers"]:
        atomic_json(out/"status.json", {"status": "blocked_preflight", "blockers": audit["blockers"]})
        raise RuntimeError("formal experiment blocked: " + "; ".join(audit["blockers"]))
    if args.seed in (101, 102):
        previous = args.output_dir.parent/"seed100/status.json"
        if not previous.exists() or json.loads(previous.read_text())["status"] != "complete":
            raise RuntimeError("seed100 must pass acceptance before seed101/102")
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/"src/kronos_continual").glob("*.py"))}
    code[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(__file__)
    config = {**PROTOCOL, "seed": args.seed, "device": args.device, "code_hashes": code,
              "inputs": audit, "frozen_components": ["Kronos", "Page1C adapter", "Page1C linear head"]}
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.resume:
        prior = json.loads((out/"run_config.json").read_text(encoding="utf-8"))
        if prior != config:
            raise AssertionError("resume changed locked configuration or inputs")
    else:
        atomic_json(out/"run_config.json", config)
    atomic_json(out/"environment.json", {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(), "threads": 1,
                "deterministic_algorithms": True, "device": args.device})
    atomic_json(out/"data_manifest.json", audit)
    history = history_pool(meta)
    atomic_csv(out/"fixed_history_pool.csv", meta.iloc[history])
    states, normalizer = build_states(args, meta, out/"state_cache")
    atomic_json(out/"offline_or_initial_config.json", {"offline_residual_training": False, "normalizer": normalizer,
                "initial": "Page1C frozen + exactly zero residual", "history_sha256": sha256(out/"fixed_history_pool.csv")})
    # Frozen base forward needs only fixed history plus test rows; no validation adaptation.
    base = np.zeros((len(meta), 10), np.float32)
    test_ids = meta.index[meta.split.eq("test")].to_numpy()
    train_ids = meta.index[meta.split.eq("train")].to_numpy()
    train_positions = pd.Series(np.arange(len(train_ids)), index=train_ids)
    base[history] = base_predict(frozen_model, args.hidden_dir/"hidden_train.npy", args.device, train_positions.loc[history].to_numpy())
    base[test_ids] = base_predict(frozen_model, args.hidden_dir/"hidden_test.npy", args.device)
    y = np.load(args.cache_dir/"y.npy", mmap_mode="r")
    engine = MARS(args.seed, args.device)
    frozen_hash = state_hash(frozen_model.state_dict())
    result = stream(engine, meta, base, states, y, history, out, args.seed, fingerprint, resume=args.resume)
    if state_hash(frozen_model.state_dict()) != frozen_hash:
        raise AssertionError("frozen Page1C was mutated")
    from .evaluation import evaluate_pair
    evaluate_pair(*result, out)
    status = json.loads((out/"status.json").read_text())
    atomic_json(out/"status.json", {**status, "status": "complete", "acceptance_passed": True, "retrospective": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=ROOT/"experiments/csi300_aligned/doubleadapt_cache_v2", help="locked label cache")
    p.add_argument("--hidden-dir", type=Path, default=ROOT/"experiments/csi300_aligned/reference/hidden_cache")
    p.add_argument("--panel-dir", type=Path, default=ROOT/"experiments/csi300_aligned/reference")
    p.add_argument("--data-root", type=Path, default=ROOT/"data/csi300_daily/csi300_daily")
    p.add_argument("--initial-checkpoint", type=Path, default=ROOT/"experiments/csi300_aligned/models/prediction_adapter/model.pt")
    p.add_argument("--expected-checkpoint-sha256", default=None,
                   help="SHA256 of selected frozen Page1-C checkpoint")
    p.add_argument("--historical-predictions", type=Path, default=None,
                   help="Historical test predictions paired with selected checkpoint")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, choices=[100, 101, 102], default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.dry_run and args.resume:
        p.error("--dry-run and --resume are mutually exclusive")
    if not args.dry_run and not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("output directory already contains artifacts; use a new directory")
    try:
        run(args)
    except Exception as exc:
        if not args.resume:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output_dir/"FAILED.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
