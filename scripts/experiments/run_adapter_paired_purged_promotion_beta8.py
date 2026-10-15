"""Beta8: strict 30-date purged gate and accepted-only low-rate promotion.

The static and continual branches are initialized from the same checkpoint and
the same locked hidden cache.  The continual branch uses matured test labels
only in a prequential update: the most recent matured dates are held out for
the gate and are excluded from replay training.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_adapter_continual_csi300_90x10 import (  # noqa: E402
    LOOKBACK,
    PRED_LEN,
    TEST_END,
    TEST_START,
    Model,
    atomic_csv,
    atomic_json,
    attach_label_end,
    date_batches,
    load_split,
    metrics,
    predict,
    rankic,
    sample_history_by_year,
    seed_everything,
    sha256_file,
    train_candidate,
    verify_boundaries,
    verify_locked_keys,
)

RECENT_DAYS = 30
MIN_RECENT_DAYS = 20
RECENT_RANKIC_DELTA = 0.002
ANCHOR_RANKIC_TOLERANCE = 0.005
RECENT_MAE_TOLERANCE = 0.05


def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--initial-checkpoint", required=True)
    p.add_argument("--expected-initial-checkpoint-sha256", required=True,
                   help="Authorized initial-checkpoint SHA256")
    p.add_argument("--reference-checkpoint", required=True,
                   help="Reference checkpoint whose parameters must equal the selected initial checkpoint")
    p.add_argument("--checkpoint-selection", default="unspecified",
                   help="Selection provenance, e.g. D_val_huber_epoch1 or D_val_rankic_epoch3")
    p.add_argument("--static-output-dir", required=True)
    p.add_argument("--continual-output-dir", required=True)
    p.add_argument('--mode', choices=['all', 'static', 'continual'], default='continual')
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-rank", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--replay-max-rows", type=int, default=120000)
    p.add_argument("--replay-recent-months", type=int, default=24)
    p.add_argument("--replay-recent-fraction", type=float, default=0.70)
    p.add_argument("--recent-validation-days", type=int, default=RECENT_DAYS)
    p.add_argument("--minimum-recent-validation-days", type=int, default=MIN_RECENT_DAYS)
    p.add_argument("--recent-rankic-improvement", type=float, default=RECENT_RANKIC_DELTA)
    p.add_argument("--anchor-rankic-tolerance", type=float, default=ANCHOR_RANKIC_TOLERANCE)
    p.add_argument("--recent-mae-tolerance", type=float, default=RECENT_MAE_TOLERANCE)
    return vars(p.parse_args())


def _ensure_clean_output(path: Path) -> None:
    if any((path / name).exists() for name in ("summary.json", "predictions.csv", "FAILED.json", 'run_config.json', 'initial_pair_audit.json')):
        raise FileExistsError(f"output directory already contains results: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _key_set(meta: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, meta[["as_of_date", "stock"]].astype(str).to_numpy()))


def _assert_same_frame_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if _key_set(left) != _key_set(right):
        raise AssertionError(f"{label} key sets differ")


def _model_predictions(model, h, y, meta, device) -> np.ndarray:
    model.eval()
    local = meta.reset_index(drop=True)
    out = np.empty((len(local), PRED_LEN), dtype=np.float32)
    with torch.no_grad():
        for ix in date_batches(local, 256, 0, False):
            source = local.iloc[ix]["_row_id"].to_numpy(dtype=np.int64)
            out[ix] = model(torch.as_tensor(np.asarray(h[source]), dtype=torch.float32, device=device)).cpu().numpy()
    return out


def _daily_standardized_mae(pred: np.ndarray, actual: np.ndarray, meta: pd.DataFrame) -> float:
    values = []
    p = np.asarray(pred, dtype=float).mean(axis=1)
    y = np.asarray(actual, dtype=float).mean(axis=1)
    local = meta.reset_index(drop=True)
    for _, g in local.groupby("as_of_date", sort=True):
        ix = g.index.to_numpy()
        if len(ix) < 3:
            continue
        ps, ys = p[ix], y[ix]
        psd, ysd = ps.std(), ys.std()
        if psd > 1e-12 and ysd > 1e-12:
            values.append(float(np.mean(np.abs((ps - ps.mean()) / psd - (ys - ys.mean()) / ysd))))
    return float(np.mean(values)) if values else float("nan")


def validation_metrics(model, h, y, meta, device) -> dict[str, float | int | str | None]:
    meta = meta.reset_index(drop=True)
    pred = _model_predictions(model, h, y, meta, device)
    actual = np.asarray(y[meta['_row_id'].to_numpy(dtype=np.int64)])
    value = {
        "rankic": float(rankic(pred, actual, meta)),
        "standardized_mae": _daily_standardized_mae(pred, actual, meta),
        "rows": int(len(meta)),
        "dates": int(meta.as_of_date.nunique()),
        "start": meta.as_of_date.min().date().isoformat() if len(meta) else None,
        "end": meta.as_of_date.max().date().isoformat() if len(meta) else None,
    }
    return value


def recent_holdout(test_m: pd.DataFrame, update_date: pd.Timestamp, days: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if days < 1:
        raise ValueError('recent validation days must be positive')
    mature = test_m[(test_m.as_of_date < update_date) & (test_m.label_end_date <= update_date)].copy()
    expected_counts = test_m.groupby('as_of_date').size()
    matured_counts = mature.groupby('as_of_date').size()
    complete_dates = matured_counts.index[matured_counts.eq(expected_counts.reindex(matured_counts.index))]
    unique_dates = sorted(complete_dates)
    hold_dates = unique_dates[-days:]
    hold = mature[mature.as_of_date.isin(hold_dates)].copy()
    train = mature[~mature.as_of_date.isin(hold_dates)].copy()
    stats = {
        "recent_validation_dates": len(hold_dates),
        "recent_validation_rows": len(hold),
        "recent_validation_start": hold.as_of_date.min().date().isoformat() if len(hold) else None,
        "recent_validation_end": hold.as_of_date.max().date().isoformat() if len(hold) else None,
        "matured_test_rows_available": len(mature),
    }
    return hold, train, stats


def select_replay_excluding(train_m: pd.DataFrame, mature_test_m: pd.DataFrame, update_date: pd.Timestamp, cfg: dict, seed: int):
    """Same replay policy as the original runner, after recent holdout removal."""
    train_pool = train_m.copy()
    train_pool["_source"] = "train"
    test_pool = mature_test_m.copy()
    test_pool["_source"] = "test"
    pool = pd.concat([train_pool, test_pool], ignore_index=True, sort=False)
    pool = pool[pool.label_end_date <= update_date].copy()
    if pool.empty:
        raise AssertionError("replay is empty after recent validation holdout")
    cutoff = update_date - pd.DateOffset(months=cfg["replay_recent_months"])
    recent = pool[pool.label_end_date >= cutoff]
    history = pool[pool.label_end_date < cutoff]
    cap = min(int(cfg["replay_max_rows"]), len(pool))
    recent_n = min(int(round(cap * cfg["replay_recent_fraction"])), len(recent))
    history_n = min(cap - recent_n, len(history))
    spare = cap - recent_n - history_n
    if spare:
        add = min(spare, len(recent) - recent_n)
        recent_n += add
        spare -= add
    if spare:
        history_n += min(spare, len(history) - history_n)
    rng = np.random.default_rng(seed)
    recent_selected = recent if recent_n >= len(recent) else recent.iloc[np.sort(rng.choice(len(recent), recent_n, replace=False))]
    historical_selected = sample_history_by_year(history, history_n, rng)
    chosen = pd.concat([recent_selected, historical_selected], ignore_index=True, sort=False)
    chosen = chosen.sort_values(["as_of_date", "stock", "_source"]).reset_index(drop=True)
    return chosen, {
        "selected_rows": len(chosen),
        "selected_train_rows": int(chosen._source.eq("train").sum()),
        "selected_test_rows": int(chosen._source.eq("test").sum()),
        "selected_recent_rows": len(recent_selected),
        "selected_historical_rows": len(historical_selected),
        "selected_max_as_of_date": chosen.as_of_date.max().date().isoformat(),
        "selected_max_label_end_date": chosen.label_end_date.max().date().isoformat(),
    }


def gate_decision(current: dict, candidate: dict, cfg: dict) -> tuple[bool, dict]:
    current_mae = float(current["recent"]["standardized_mae"])
    candidate_mae = float(candidate["recent"]["standardized_mae"])
    recent_delta = float(candidate["recent"]["rankic"] - current["recent"]["rankic"])
    anchor_delta = float(candidate["anchor"]["rankic"] - current["anchor"]["rankic"])
    mae_ratio = candidate_mae / current_mae if current_mae > 0 else float("inf")
    valid = bool(np.isfinite([current_mae, candidate_mae, recent_delta, anchor_delta, mae_ratio]).all())
    recent_ok = valid and recent_delta >= cfg["recent_rankic_improvement"] - 1e-12
    anchor_ok = valid and anchor_delta >= -cfg["anchor_rankic_tolerance"] - 1e-12
    mae_ok = valid and mae_ratio <= 1.0 + cfg["recent_mae_tolerance"] + 1e-12
    return bool(recent_ok and anchor_ok and mae_ok), {
        "current_recent_rankic": current["recent"]["rankic"],
        "candidate_recent_rankic": candidate["recent"]["rankic"],
        "recent_rankic_delta": recent_delta,
        "current_recent_standardized_mae": current_mae,
        "candidate_recent_standardized_mae": candidate_mae,
        "recent_mae_ratio": mae_ratio,
        "current_anchor_rankic": current["anchor"]["rankic"],
        "candidate_anchor_rankic": candidate["anchor"]["rankic"],
        "anchor_rankic_delta": anchor_delta,
        "recent_rankic_gate_pass": recent_ok,
        "anchor_rankic_gate_pass": anchor_ok,
        "recent_mae_gate_pass": mae_ok,
    }


def verify_checkpoint_hash(actual: str, expected: str) -> None:
    expected = expected.strip().lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected initial checkpoint SHA256 must be 64 hexadecimal characters')
    if actual.lower() != expected:
        raise AssertionError(
            'D checkpoint SHA256 does not match the authorized current hash: '
            f'expected={expected}, actual={actual}'
        )


def _hashes(cache: Path, panel: Path, checkpoint: Path, script: Path) -> dict[str, str]:
    names = ["hidden_train.npy", "hidden_val.npy", "hidden_test.npy", "labels_train.npy", "labels_val.npy", "labels_test.npy", "meta_train.csv", "meta_val.csv", "meta_test.csv"]
    result = {name: sha256_file(cache / name) for name in names}
    result["eligible_stock_date_pairs.csv"] = sha256_file(panel / "eligible_stock_date_pairs.csv")
    result["date_windows.csv"] = sha256_file(panel / "date_windows.csv")
    result["initial_checkpoint"] = sha256_file(checkpoint)
    result["runner"] = sha256_file(script)
    return result


def _save_static(model, test_h, test_y, test_m, device, out: Path, cfg: dict, hashes: dict) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "version_000.pt")
    pred = pd.DataFrame(predict(model, test_h, test_y, test_m, device, 0, False, "adapter_dpaired_static", cfg["seed"]))
    atomic_csv(out / "predictions.csv", pred)
    atomic_csv(out / "daily_metrics.csv", metrics(pred))
    atomic_json(out / "run_config.json", {**cfg, "branch": "paired_static", "hashes": hashes, 'test_start':str(TEST_START.date()),'test_end':str(TEST_END.date()), 'test_rows':len(pred), 'test_dates':int(pred.as_of_date.nunique()),'historical_identity_check':'authorized_selection_checkpoint_sha256_matches_and_state_dict_equals_reference', 'historical_checkpoint_sha256_available':False, 'current_checkpoint_authorized_by_user':True})
    atomic_json(out / "summary.json", {"status": "complete", "branch": "paired_static", "test_dates": int(pred.as_of_date.nunique()), "test_rows": len(pred), "version": 0})
    return pred


def paired_comparison(static_pred: pd.DataFrame, continual_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    key = ["as_of_date", "stock"]
    left = static_pred.sort_values(key).reset_index(drop=True)
    right = continual_pred.sort_values(key).reset_index(drop=True)
    if left[key].astype(str).to_records(index=False).tolist() != right[key].astype(str).to_records(index=False).tolist():
        raise AssertionError("paired static/continual prediction keys differ")
    pred_cols = ["pred_signal", "pred_endpoint_return"] + [f"pred_return_{i}" for i in range(1, PRED_LEN + 1)]
    out = left[key + pred_cols].copy()
    summary = {}
    for col in pred_cols:
        diff = right[col].to_numpy(dtype=float) - left[col].to_numpy(dtype=float)
        out[f"continual_minus_static_{col}"] = diff
        summary[f"initial_or_final_max_abs_diff_{col}"] = float(np.max(np.abs(diff))) if len(diff) else 0.0
    return out, summary


PROTOCOL_METADATA = {
    'recent_validation_days': 30,
    'minimum_recent_validation_days': 20,
    'promotion_policy': 'after_candidate_gate_acceptance_only',
    'promotion_epochs': 1,
    'promotion_head_lr': 3e-5,
    'promotion_adapter_lr': 1e-5,
    'promotion_loss': 'Huber + 0.2 pairwise ranking',
    'promotion_history_definition': 'all previously selected purged replay rows, unchanged sample selection; append all recent gate rows',
    'promotion_epoch_selection': 'none; exactly one epoch; no validation used',
    'candidate_epoch_selection': 'fixed anchor only; recent gate evaluated only after training',
    'current_checkpoint_authorized_by_user': True,
    'historical_checkpoint_sha256_available': False,
    'historical_sha256_status': 'not_independently_verified; user authorized current checkpoint hash',
    'future20_evaluation_excludes_gate_rows': True,
}


def enforce_protocol(cfg):
    expected = {'recent_validation_days': 30, 'minimum_recent_validation_days': 20,
                'seed': 100, 'update_epochs': 2, 'batch_size': 16,
                'head_lr': 3e-4, 'adapter_lr': 1e-4, 'lambda_rank': .2,
                'recent_rankic_improvement': .002, 'anchor_rankic_tolerance': .005,
                'recent_mae_tolerance': .05, 'weight_decay':1e-4, 'patience':2,
                'replay_recent_months':24, 'replay_recent_fraction':.7, 'replay_max_rows':120000}
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'beta8 preregistered parameter changed: {key}={cfg[key]!r}, expected {value!r}')


def attach_label_interval(meta, panel):
    windows = pd.read_csv(panel / 'date_windows.csv')
    for col in ['as_of_date', 'label_start_date', 'label_end_date']:
        windows[col] = pd.to_datetime(windows[col], errors='raise').dt.normalize()
    out = meta.drop(columns=['label_start_date','label_end_date'], errors='ignore').merge(
        windows[['as_of_date','label_start_date','label_end_date']].drop_duplicates('as_of_date'),
        on='as_of_date', how='left', validate='many_to_one')
    if out[['label_start_date','label_end_date']].isna().any().any():
        raise AssertionError('missing label interval from locked date_windows')
    return out


def enough_gate_days(stats, cfg):
    return stats['recent_validation_dates'] >= cfg['minimum_recent_validation_days']


def prediction_stream(dates, test_h, test_y, test_m, device, supplier, seed):
    """Yield today's predictions before caller updates; refresh model on next resume."""
    for pos, date in enumerate(dates):
        model, version = supplier()
        dm = test_m[test_m.as_of_date.eq(date)].copy()
        yield pos, date, predict(model, test_h, test_y, dm, device, version, False, 'adapter_dpaired_continual', seed)


def fit_candidate_anchor_only(candidate, uh, uy, um, val_h, val_y, val_m, cfg, device, seed):
    # This interface deliberately has no recent-gate arrays or labels.
    return train_candidate(candidate, uh, uy, um, val_h, val_y, val_m, cfg, device, seed, cfg['update_epochs'])


def select_purged_replay(train_m, mature_test_m, gate_m, update_date, cfg, seed):
    if gate_m.empty:
        raise ValueError('cannot construct purged replay without gate samples')
    boundary = gate_m.label_start_date.min()
    mature_test_m = mature_test_m[mature_test_m.label_end_date < boundary].copy()
    train_m = train_m[train_m.label_end_date < boundary].copy()
    chosen, stats = select_replay_excluding(train_m, mature_test_m, update_date, cfg, seed)
    stats.update({'purge_boundary_label_start_date': boundary.date().isoformat(),
                  'purge_train_max_label_end_date': chosen.label_end_date.max().date().isoformat() if len(chosen) else None,
                  'purge_applied': True})
    verify_purged(chosen, gate_m)
    return chosen, stats


def verify_purged(train, gate):
    if not len(gate):
        return
    boundary = gate.label_start_date.min()
    if not (train.label_end_date < boundary).all():
        raise AssertionError('replay label_end_date must be strictly earlier than gate label_start_date')
    keys = set(zip(train.as_of_date.astype(str), train.stock.astype(str)))
    gate_keys = set(zip(gate.as_of_date.astype(str), gate.stock.astype(str)))
    if keys & gate_keys:
        raise AssertionError('gate keys leaked into replay')


def deploy_after_gate(accepted, current, candidate, chosen, gate, train_h, train_y,
                      test_h, test_y, cfg, device, seed, update_date):
    detail = {'promotion_ran': False, 'promotion_epochs': 1, 'promotion_head_lr': 3e-5,
              'promotion_adapter_lr': 1e-5, 'promotion_gate_rows': 0, 'promotion_history_rows': 0}
    if not accepted:
        detail['promotion_decision'] = 'skipped_candidate_rejected'
        return current, detail
    history = chosen.copy()
    gate_ids = gate._row_id.to_numpy(dtype=np.int64)
    hist_train = history[history._source.eq('train')]
    hist_test = history[history._source.eq('test')]
    hparts, yparts, mparts = [], [], []
    if len(hist_train):
        ids = hist_train._row_id.to_numpy(dtype=np.int64)
        hparts.append(np.asarray(train_h[ids])); yparts.append(np.asarray(train_y[ids])); mparts.append(hist_train)
    if len(hist_test):
        ids = hist_test._row_id.to_numpy(dtype=np.int64)
        hparts.append(np.asarray(test_h[ids])); yparts.append(np.asarray(test_y[ids])); mparts.append(hist_test)
    if len(gate):
        hparts.append(np.asarray(test_h[gate_ids])); yparts.append(np.asarray(test_y[gate_ids])); mparts.append(gate.assign(_source='test'))
    promotion_m = pd.concat(mparts, ignore_index=True, sort=False).sort_values(['as_of_date','stock']).reset_index(drop=True)
    promotion_h = np.concatenate(hparts); promotion_y = np.concatenate(yparts)
    unsorted_m = pd.concat(mparts, ignore_index=True, sort=False)
    order = unsorted_m.sort_values(['as_of_date','stock']).index.to_numpy()
    promotion_h = promotion_h[order]; promotion_y = promotion_y[order]
    if promotion_m[['_source','_row_id']].duplicated().any():
        raise AssertionError('duplicate rows in promotion training set')
    p_cfg = dict(cfg, head_lr=3e-5, adapter_lr=1e-5, update_epochs=1, patience=1)
    promoted = copy.deepcopy(candidate).to(device)
    history_log = train_promotion_epoch(promoted, promotion_h, promotion_y, promotion_m, p_cfg, device, seed)
    detail.update({'promotion_ran': True, 'promotion_decision':'accepted_and_promoted',
        'promotion_gate_rows':len(gate), 'promotion_history_rows':len(history),
        'promotion_train_rows':len(promotion_m), 'promotion_best_epoch':1,
        'promotion_history':history_log, 'promotion_lr_head':3e-5, 'promotion_lr_adapter':1e-5,
        'candidate_gate_max_label_end_date': chosen.label_end_date.max().date().isoformat() if len(chosen) else None,
        'promotion_max_label_end_date': promotion_m.label_end_date.max().date().isoformat(),
        'deployment_effective_date':None})
    return promoted, detail


def train_promotion_epoch(model, h, y, meta, cfg, device, seed):
    """Exactly one epoch, no validation, epoch selection, gate or extra tuning."""
    seed_everything(seed)
    optimizer = torch.optim.AdamW([
        {'params':model.head.parameters(), 'lr':cfg['head_lr']},
        {'params':model.adapter.parameters(), 'lr':cfg['adapter_lr']}], weight_decay=cfg['weight_decay'])
    model.train(); losses = []; ranks = []
    for ix in date_batches(meta, cfg['batch_size'], seed+1, True):
        inputs = torch.as_tensor(np.asarray(h[ix]), dtype=torch.float32, device=device)
        targets = torch.as_tensor(np.asarray(y[ix]), dtype=torch.float32, device=device)
        prediction = model(inputs)
        huber = torch.nn.functional.smooth_l1_loss(prediction, targets)
        ps, ys = prediction.mean(1), targets.mean(1)
        dp, dy = ps[:,None]-ps[None,:], ys[:,None]-ys[None,:]
        mask = dy != 0
        ranking = torch.nn.functional.softplus(-torch.sign(dy[mask])*dp[mask]).mean() if bool(mask.any()) else ps.sum()*0
        loss = huber + cfg['lambda_rank']*ranking
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(huber.detach().cpu())); ranks.append(float(ranking.detach().cpu()))
    row = {'epoch':1, 'train_huber':float(np.mean(losses)), 'train_rank_loss':float(np.mean(ranks)), 'validation_used':False}
    print('[PROMOTION]', json.dumps(row), flush=True)
    return [row]


def write_future20_predictions(out, audits, dates, test_h, test_y, test_m, device, cfg):
    pieces = []
    for item in audits:
        checkpoint = out / 'checkpoints' / f"version_{item['model_version']:03d}.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model = Model(512, PRED_LEN, 64, .1).to(device); model.load_state_dict(state); model.requires_grad_(False)
        start = list(dates).index(pd.Timestamp(item['next_test_date']))
        selected = set(dates[start:start+20])
        # Each window is its own prospective diagnostic; overlap is explicitly recorded.
        subset = test_m[test_m.as_of_date.isin(selected)].copy()
        frame = pd.DataFrame(predict(model, test_h, test_y, subset, device, item['model_version'], False,
                                     f"beta8_future20_v{item['model_version']:03d}", cfg['seed']))
        frame['trigger_update_date'] = item['update_date']; frame['window_length_trading_days'] = len(selected)
        frame['window_start'] = item['next_test_date']; frame['window_end'] = max(selected).date().isoformat()
        pieces.append(frame)
    if pieces:
        atomic_csv(out / 'accepted_update_future20_predictions.csv', pd.concat(pieces, ignore_index=True))



def main() -> None:
    cfg = parse_args()
    enforce_protocol(cfg)
    seed_everything(cfg['seed'])
    cache, panel = Path(cfg["cache_dir"]), Path(cfg["panel_dir"])
    checkpoint = Path(cfg["initial_checkpoint"])
    static_out, continual_out = Path(cfg["static_output_dir"]), Path(cfg["continual_output_dir"])
    if static_out.resolve() == continual_out.resolve():
        raise ValueError('static and continual output directories must differ')
    if cfg['mode'] != 'continual':
        _ensure_clean_output(static_out)
    if cfg['mode'] != 'static':
        _ensure_clean_output(continual_out)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D checkpoint not found: {checkpoint}")
    if cfg["batch_size"] != 16:
        raise ValueError("paired experiment requires batch_size=16")
    hashes = _hashes(cache, panel, checkpoint, Path(__file__))
    verify_checkpoint_hash(hashes['initial_checkpoint'], cfg['expected_initial_checkpoint_sha256'])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")
    train_h, train_y, train_m = load_split(cache, "train")
    val_h, val_y, val_m = load_split(cache, "val")
    test_h, test_y, test_m = load_split(cache, "test")
    train_m, val_m, test_m = (attach_label_interval(train_m, panel), attach_label_interval(val_m, panel), attach_label_interval(test_m, panel))
    verify_locked_keys(train_m, panel, "train", True)
    verify_locked_keys(val_m, panel, "val", True)
    verify_locked_keys(test_m, panel, "test", True)
    verify_boundaries(train_m, val_m, test_m, True)
    if len(test_m) != 66103 or test_m.as_of_date.nunique() != 226:
        raise AssertionError(f"locked test shape mismatch: rows={len(test_m)}, dates={test_m.as_of_date.nunique()}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = Path(cfg['reference_checkpoint'])
    reference_state = torch.load(reference, map_location='cpu', weights_only=True)
    if state.keys() != reference_state.keys() or any(not torch.equal(state[k], reference_state[k]) for k in state):
        raise AssertionError('selected initial checkpoint parameters differ from reference checkpoint')
    hashes['reference_checkpoint'] = sha256_file(reference)
    environment = {'python': platform.python_version(), 'pytorch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device)}
    hashes['paired_utilities_runner'] = sha256_file(EXPERIMENT_DIR / 'run_adapter_paired_dualgate_csi300_90x10.py')
    hashes['training_runner'] = sha256_file(EXPERIMENT_DIR / 'run_adapter_continual_csi300_90x10.py')
    outputs = [static_out] if cfg['mode']=='static' else [continual_out] if cfg['mode']=='continual' else [static_out,continual_out]
    for output in outputs:
        atomic_json(output / 'environment.json', environment)
        atomic_json(output / 'run_config.json', {**cfg, **PROTOCOL_METADATA, 'hashes': hashes, 'initial_checkpoint_sha256': hashes['initial_checkpoint'], 'historical_identity_check': 'authorized_selection_checkpoint_sha256_matches_and_state_dict_equals_reference', 'historical_checkpoint_sha256_available': False, 'current_checkpoint_authorized_by_user': True, 'recent_validation_excluded_from_replay': True})
    static_model = Model(512, PRED_LEN, 64, 0.1).to(device)
    static_model.load_state_dict(state, strict=True)
    static_model.requires_grad_(False)
    if cfg['mode'] == 'continual':
        static_config = json.loads((static_out/'run_config.json').read_text(encoding='utf-8'))
        for name in [k for k in hashes if k not in ('runner','training_runner','paired_utilities_runner')]:
            if static_config['hashes'].get(name) != hashes[name]:
                raise AssertionError(f'cached Static identity differs: {name}')
        static_pred = pd.read_csv(static_out/'predictions.csv', dtype={'stock':str})
    else:
        static_pred = _save_static(static_model, test_h, test_y, test_m, device, static_out, {**cfg, "initial_checkpoint_sha256": hashes["initial_checkpoint"]}, hashes)
    if cfg['mode']=='static':
        return
    continual_model = Model(512, PRED_LEN, 64, 0.1).to(device)
    continual_model.load_state_dict(state, strict=True)
    initial_continual_pred = pd.DataFrame(predict(continual_model, test_h, test_y, test_m, device, 0, False, "adapter_dpaired_continual", cfg["seed"]))
    # Reused Static rows must preserve exact prediction order and all labels.
    if not static_pred[['as_of_date','stock']].astype(str).equals(initial_continual_pred[['as_of_date','stock']].astype(str)):
        raise AssertionError('static/continual ordered keys differ')
    for col in [f'actual_return_{i}' for i in range(1,11)]+['actual_signal','actual_endpoint_return','label_end_date']:
        if col=='label_end_date':
            if not static_pred[col].astype(str).equals(initial_continual_pred[col].astype(str)):
                raise AssertionError('static/continual label dates differ')
        elif not np.allclose(static_pred[col],initial_continual_pred[col],atol=1e-12,rtol=0):
            raise AssertionError(f'static/continual actual labels differ: {col}')
    for col in ["pred_signal", "pred_endpoint_return"] + [f"pred_return_{i}" for i in range(1, PRED_LEN + 1)]:
        if not np.allclose(static_pred[col].to_numpy(), initial_continual_pred[col].to_numpy(), atol=1e-6, rtol=0):
            raise AssertionError(f"paired version_000 prediction mismatch in {col}")
    init_max = float(np.max(np.abs(static_pred[[f'pred_return_{i}' for i in range(1,11)]].to_numpy()-initial_continual_pred[[f'pred_return_{i}' for i in range(1,11)]].to_numpy())))
    atomic_json(continual_out / 'initial_pair_audit.json', {'status':'passed', 'rows':len(test_m), 'dates':226, 'max_abs_diff':init_max, 'checkpoint_parameters_match_reference':True})
    dates = pd.DatetimeIndex(sorted(test_m.as_of_date.unique()))
    update_dates = [dates[i] for i in range(len(dates) - 1) if dates[i].month != dates[i + 1].month]
    update_set = set(update_dates)
    anchor_current = validation_metrics(continual_model, val_h, val_y, val_m, device)
    rows, logs, versions = [], [], 0
    (continual_out / "checkpoints").mkdir(exist_ok=True)
    torch.save(state, continual_out / "checkpoints" / "version_000.pt")
    seen_mature: set[int] = set()
    started = time.time()
    train_cfg = dict(cfg)
    future_audits = []
    for pos, date, day_rows in prediction_stream(dates, test_h, test_y, test_m, device,
                                                lambda: (continual_model, versions), cfg['seed']):
        rows.extend(day_rows)
        if date not in update_set:
            continue
        recent_m, old_mature_m, recent_stats = recent_holdout(test_m, date, cfg["recent_validation_days"])
        available_ids = set(test_m[(test_m.as_of_date < date) & (test_m.label_end_date <= date)]["_row_id"].astype(int))
        new_ids = available_ids - seen_mature
        seen_mature.update(available_ids)
        base_log = {"update_date": date.date().isoformat(), "model_version_before": versions, "model_version_after": versions, **recent_stats,
                    "recent_rankic_threshold": cfg["recent_rankic_improvement"], "anchor_rankic_tolerance": cfg["anchor_rankic_tolerance"], "recent_mae_tolerance": cfg["recent_mae_tolerance"], "new_matured_test_rows": len(new_ids), "next_test_date": dates[pos + 1].date().isoformat() if pos + 1 < len(dates) else None}
        if not enough_gate_days(recent_stats, cfg):
            skipped = {field:None for field in ['current_recent_rankic','candidate_recent_rankic','recent_rankic_delta','current_recent_standardized_mae','candidate_recent_standardized_mae','recent_mae_ratio','candidate_anchor_rankic','anchor_rankic_delta','selected_max_label_end_date','best_epoch']}
            logs.append({**base_log, **skipped, 'current_anchor_rankic':anchor_current['rankic'], 'selected_replay_rows':0,'selected_train_rows':0,'selected_test_rows':0,"accepted": False, "decision": "skipped_insufficient_recent_validation", "promotion_ran":False, "promotion_epochs":1, "promotion_head_lr":3e-5, "promotion_adapter_lr":1e-5, "promotion_gate_rows":0, "promotion_history_rows":0, "history": []})
            atomic_csv(continual_out/'update_log.csv',pd.DataFrame(logs))
            continue
        chosen, replay_stats = select_purged_replay(train_m, old_mature_m, recent_m, date, train_cfg, cfg["seed"] + pos)
        train_part, test_part = chosen[chosen._source.eq("train")], chosen[chosen._source.eq("test")]
        train_ids, test_ids = train_part._row_id.to_numpy(dtype=np.int64), test_part._row_id.to_numpy(dtype=np.int64)
        uh = np.concatenate([np.asarray(train_h[train_ids]), np.asarray(test_h[test_ids])])
        uy = np.concatenate([np.asarray(train_y[train_ids]), np.asarray(test_y[test_ids])])
        um = pd.concat([train_part, test_part], ignore_index=True)
        if _key_set(um) & _key_set(recent_m) or _key_set(um) & _key_set(val_m):
            raise AssertionError('validation samples entered replay')
        if (um.label_end_date > date).any() or (test_part.as_of_date >= date).any():
            raise AssertionError('future sample entered replay')
        verify_purged(um, recent_m)
        atomic_csv(continual_out / 'replay_audit' / f'{date.date()}.csv', chosen[['as_of_date','stock','label_start_date','label_end_date','_source','_row_id']])
        atomic_csv(continual_out / 'gate_audit' / f'{date.date()}.csv', recent_m[['as_of_date','stock','label_start_date','label_end_date','_row_id']])
        candidate = copy.deepcopy(continual_model).to(device)
        seed_everything(cfg['seed']+pos+1)
        candidate, history, _, best_epoch = fit_candidate_anchor_only(candidate, uh, uy, um, val_h, val_y, val_m, train_cfg, device, cfg["seed"] + pos + 1)
        current_recent = validation_metrics(continual_model, test_h, test_y, recent_m, device)
        candidate_recent = validation_metrics(candidate, test_h, test_y, recent_m, device)
        candidate_anchor = validation_metrics(candidate, val_h, val_y, val_m, device)
        accepted, gate = gate_decision({"recent": current_recent, "anchor": anchor_current}, {"recent": candidate_recent, "anchor": candidate_anchor}, cfg)
        promotion_log = {'promotion_ran':False, 'promotion_epochs':1, 'promotion_head_lr':3e-5, 'promotion_adapter_lr':1e-5, 'promotion_gate_rows':0, 'promotion_history_rows':0,
                         'candidate_gate_checkpoint':None, 'deployed_checkpoint':None, 'deployment_effective_date':None}
        if accepted:
            torch.save(candidate.state_dict(), continual_out / "checkpoints" / f"candidate_gate_{date.date()}.pt")
        continual_model, promotion_details = deploy_after_gate(accepted, continual_model, candidate, chosen, recent_m,
                train_h, train_y, test_h, test_y, cfg, device, cfg['seed'] + pos + 10000, date)
        promotion_log.update(promotion_details)
        if accepted:
            versions += 1
            anchor_current = validation_metrics(continual_model, val_h, val_y, val_m, device)
            torch.save(continual_model.state_dict(), continual_out / "checkpoints" / f"version_{versions:03d}.pt")
            promotion_log.update(candidate_gate_checkpoint=f'checkpoints/candidate_gate_{date.date()}.pt',
                         deployed_checkpoint=f'checkpoints/version_{versions:03d}.pt', deployment_effective_date=base_log['next_test_date'])
            future_audits.append({'update_date':base_log['update_date'], 'model_version':versions, 'next_test_date':base_log['next_test_date']})
        logs.append({**base_log, **replay_stats, **gate, **promotion_log, 'selected_replay_rows':replay_stats['selected_rows'], "accepted": accepted, "decision": "accepted_then_promoted" if accepted else "rejected_dual_validation_gate", "model_version_after": versions, "selected_max_as_of_date": replay_stats["selected_max_as_of_date"], "selected_max_label_end_date": replay_stats["selected_max_label_end_date"], "best_epoch": best_epoch, "history": history})
        print('[UPDATE]',json.dumps({'date':base_log['update_date'],'accepted':accepted,'version':versions,
            'recent_rankic_delta':gate['recent_rankic_delta'],'anchor_rankic_delta':gate['anchor_rankic_delta'],
            'promotion_ran':promotion_log['promotion_ran']},default=str),flush=True)
        atomic_csv(continual_out / 'update_log.csv', pd.DataFrame(logs))
        atomic_json(continual_out / 'progress.json', {'date':str(date.date()), 'updates':len(logs), 'version':versions})
    out_pred = pd.DataFrame(rows)
    out_pred.loc[out_pred.as_of_date.isin([d.date().isoformat() for d in update_dates]), "update_ran_after_prediction"] = True
    (continual_out / "checkpoints").mkdir(exist_ok=True)
    atomic_csv(continual_out / "predictions.csv", out_pred)
    atomic_csv(continual_out / "daily_metrics.csv", metrics(out_pred))
    atomic_csv(continual_out / "update_log.csv", pd.DataFrame(logs))
    write_future20_predictions(continual_out, future_audits, dates, test_h, test_y, test_m, device, cfg)
    pair_frame, pair_summary = paired_comparison(static_pred, out_pred)
    before = out_pred.model_version.eq(0)
    cols = [f'pred_return_{i}' for i in range(1,11)]
    if not np.allclose(out_pred.loc[before,cols].to_numpy(), static_pred.loc[before,cols].to_numpy(),atol=1e-6,rtol=0):
        raise AssertionError('predictions changed before the first accepted update')
    atomic_csv(continual_out / "paired_prediction_comparison.csv", pair_frame)
    config = {**cfg, **PROTOCOL_METADATA, "branch": "paired_continual_purged_promotion_beta8", "initial_checkpoint_sha256": hashes["initial_checkpoint"], 'paired_static_checkpoint':str(static_out/'checkpoints'/'version_000.pt'), 'historical_checkpoint_sha256_available':False, 'current_checkpoint_authorized_by_user':True, 'historical_identity_check':'authorized_selection_checkpoint_sha256_matches_and_state_dict_equals_reference', "paired_static_output": str(static_out), "recent_validation_excluded_from_replay": True, "hashes": hashes, "device_resolved": str(device), "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()), "test_dates": len(dates), "test_rows": len(out_pred)}
    atomic_json(continual_out / "run_config.json", config)
    atomic_json(continual_out / "summary.json", {"status": "complete", "branch": "paired_continual_purged_promotion_beta8", "test_dates": len(dates), "test_rows": len(out_pred), "updates": len(logs), "accepted_updates": int(sum(bool(x["accepted"]) for x in logs)), "final_version": versions, "elapsed_seconds": time.time() - started, "paired_prediction_summary": pair_summary})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not overwrite existing experiments on a refusal to reuse outputs.
        args = parse_args()
        output = Path(args['continual_output_dir'])
        if output.exists() and not (output/'summary.json').exists() and not isinstance(error,FileExistsError):
            atomic_json(output/'FAILED.json',{'status':'failed','error':repr(error),'traceback':traceback.format_exc()})
        raise
