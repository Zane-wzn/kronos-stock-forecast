"""Load the archived Adapter and replay its original continual updates."""
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / ".peer_pca_inputs"
RESULTS = ROOT / "last_try" / "runs"
CONTROLS = INPUTS / "controls_120d"
SEEDS = (100, 101, 102)
KEYS = ["as_of_date", "stock"]
PC = [f"pred_return_{i}" for i in range(1, 11)]
AC = [f"actual_return_{i}" for i in range(1, 11)]
PRED_LEN = 10
EXPECTED = {
    "csi300": (226, 66103, 216, 69120),
    "nasdaq100": (235, 23030, 225, 220500),
    "sp500": (235, 117464, 225, 1124540),
}


def config(market):
    top_k, drop_n = {"csi300": (30, 3), "nasdaq100": (10, 1), "sp500": (50, 5)}[market]
    directory = INPUTS / market
    return dict(market=market, cache=directory / "hidden_cache", home=directory,
                archive=directory / "native", top_k=top_k, drop_n=drop_n, min_hold=5)


def new_model():
    model = Model()
    nn.init.zeros_(model.adapter.up.weight)
    nn.init.zeros_(model.adapter.up.bias)
    return model



class Adapter(nn.Module):
    def __init__(self, d: int = 512, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d)

    def forward(self, h):
        return h + self.up(self.drop(F.gelu(self.down(self.ln(h)))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.head = nn.Linear(512, PRED_LEN)

    def forward(self, h):
        return self.head(self.adapter(h))


def make_model(checkpoint: Path, update_head: bool):
    model = Model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if update_head:
        for p in model.head.parameters():
            p.requires_grad_(True)
    return model


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensor(array, device):
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)


def predict_tensor(model, hidden, batch=256):
    model.eval()
    with torch.inference_mode():
        values = [model(hidden[i:i + batch]).cpu().numpy() for i in range(0, len(hidden), batch)]
    return np.concatenate(values)


def make_schedule(meta):
    # Selection depends only on metadata; numeric labels remain behind a maturity check.
    dates = sorted(meta.as_of_date.unique())
    return {d: meta.index[(meta.label_end_date.eq(d)) & meta.as_of_date.lt(d)].to_numpy()
            for d in dates}


def read_mature(labels, meta, ids, today):
    selected = meta.iloc[ids]
    if not (selected.label_end_date.le(today).all() and selected.as_of_date.lt(today).all()):
        raise AssertionError("unreleased complete-path label access")
    targets = np.asarray(labels[ids])
    if not np.isfinite(targets).all(): raise FloatingPointError("mature labels nonfinite")
    return targets


def online_optimizer(model, cfg):
    c = cfg["online"]
    return torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": c["adapter_lr"]},
        {"params": model.head.parameters(), "lr": c["head_lr"]}], weight_decay=0)


def daily_update(model, optimizer, hidden, targets, micro_batch=256):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    n = len(targets)
    total = 0.
    for start in range(0, n, micro_batch):
        target = targets[start:start + micro_batch]
        loss_sum = F.smooth_l1_loss(model(hidden[start:start + micro_batch]), target,
                                   beta=1, reduction="sum")
        (loss_sum / targets.numel()).backward()
        total += float(loss_sum.detach())
    norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True))
    optimizer.step()
    return {"loss": total / targets.numel(), "grad_norm": norm,
            "unique_feature_rows": n, "supervision_scalars": targets.numel(), "optimizer_steps": 1}


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def align_predictions(meta,frame,columns=PC):
    assert len(meta)==len(frame) and not frame.duplicated(KEYS).any()
    joined=meta[KEYS].merge(frame[KEYS+columns],on=KEYS,how='left',sort=False,validate='one_to_one')
    assert joined[KEYS].equals(meta[KEYS].reset_index(drop=True))
    result=joined[columns].to_numpy(dtype=np.float64)
    assert np.isfinite(result).all()
    return result


def verify_support(schedule,original):
    columns=['update_date','origin_date','stock','horizon','complete_date']
    expected=pd.DataFrame([dict(update_date=d,**e) for d,events in schedule.items() for e in events])
    a=expected[columns].sort_values(columns).reset_index(drop=True)
    b=original[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    return dict(status='passed',matched_scalars=len(a),row_ids_matched_by_stock_origin=True)


def native_schedule(cfg,meta):
    """Reconstruct original supports and compare every scalar to original journals."""
    original=pd.read_csv(cfg['access'],dtype={'stock':str})
    if 'horizon' not in original:
        original=original.assign(horizon=original.horizons.str.split('|')).explode('horizon')
        original['horizon']=original.horizon.astype(int)
    keys={(r.as_of_date,r.stock):int(i) for i,r in meta.iterrows()}
    native=original.copy();native['row_id']=[keys[(r.origin_date,r.stock)] for r in native.itertuples()]
    dates=sorted(meta.as_of_date.unique());groups={d:g for d,g in meta.groupby('as_of_date',sort=True)}
    byend={d:g for d,g in meta.groupby('label_end_date',sort=True)}
    schedule={d:[] for d in dates}
    if cfg['market']=='csi300':
        training=pd.read_csv(cfg['cache']/'meta_train.csv',usecols=['stock'],dtype=str)
        key=lambda stock:hashlib.sha256(('prefix-pilot-v1|'+str(stock)).encode()).hexdigest()
        pool=set(sorted(set(training.stock),key=key)[:60])
        label_dates={d:g.label_dates.iloc[0].split('|') for d,g in groups.items()}
        training_by_date={d:g[g.stock.isin(pool)] for d,g in groups.items()}
        for today in dates[10:]:
            complete=[d for d in dates if d<today and label_dates[d][-1]==today]
            if not complete:continue
            origin=complete[-1];frows=training_by_date[origin]
            for k in range(10):
                prefix=[d for d in dates if d<today and label_dates[d][k]==today]
                if not prefix:continue
                prows=training_by_date[prefix[-1]]
                stocks=sorted(set(frows.stock)&set(prows.stock),key=key)[:32]
                lookup=frows.set_index('stock')
                for stock in stocks:
                    i=keys[(origin,stock)]
                    schedule[today].append(dict(row_id=i,stock=stock,origin_date=origin,horizon=k+1,
                        label_date=label_dates[origin][k],complete_date=label_dates[origin][-1]))
    else:
        for today in dates:
            if today not in byend:continue
            for i,r in byend[today].iterrows():
                assert r.as_of_date<today
                for k in range(1,11):schedule[today].append(dict(row_id=int(i),stock=r.stock,
                    origin_date=r.as_of_date,horizon=k,complete_date=r.label_end_date))
    audit=verify_support(schedule,native)
    return schedule,audit


def panel(cache, split):
    meta = pd.read_csv(cache / f"meta_{split}.csv", dtype=str)
    hidden = np.load(cache / f"hidden_{split}.npy", mmap_mode="r")
    assert hidden.shape == (len(meta), 512) and not meta.duplicated(KEYS).any()
    assert meta.input_end_date.eq(meta.as_of_date).all()
    return meta, hidden


def supervision(market, cfg, meta, seed):
    native_root = cfg["archive"] / f"seed{seed}"
    if market == "csi300":
        native_path = native_root / "A/A/access_log.csv"
        checkpoint = native_root / "offline/model.pt"
        expected_hash = read_json(native_root / "offline/complete.json")["sha256"]
        encoder = make_model(checkpoint, False).adapter
    else:
        native_path = native_root / "Adapter_CL/supervision_access.csv.gz"
        checkpoint = native_root / "offline/initial.pt"
        expected_hash = read_json(native_root / "offline/complete.json")["initial_sha256"]
        model = new_model().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        encoder = model.adapter
    assert sha(checkpoint) == expected_hash
    events, _ = native_schedule(dict(cfg, access=native_path), meta)
    schedule = {
        date: (np.array([e["row_id"] for e in batch]), np.array([e["horizon"] - 1 for e in batch]))
        for date, batch in events.items() if batch
    }
    assert (len(schedule), sum(len(rows) for rows, _ in schedule.values())) == EXPECTED[market][2:]
    return schedule, encoder, dict(checkpoint=str(checkpoint), checkpoint_sha256=expected_hash,
                                    access=str(native_path), access_sha256=sha(native_path))


def capture_daily(adapter, hidden, days, advance, device, *, observe=None):
    """Encode before consuming any of that day's new feedback."""
    features = np.empty(hidden.shape, dtype=np.float32)
    versions = []
    version = 0
    adapter.eval()
    for date, ids in days.items():
        batch = tensor(hidden[ids], device)
        with torch.no_grad():
            encoded = adapter(batch)
            features[ids] = encoded.cpu().numpy()
            if observe is not None:
                observe(date, ids, encoded)
        versions.append(version)
        version += int(advance(date, ids, batch, encoded, version))
    return features, versions


def baseline_embeddings(market, cfg, checkpoint, encoder, test, hidden, labels, base, schedule, device, *, observe=None):
    native = checkpoint.parent.parent
    days = test.groupby("as_of_date", sort=True).indices
    errors = []
    state_error = None
    if market == "csi300":
        model = make_model(checkpoint, False)
        archive = native / "A/checkpoints/A"
        assert {p.stem for p in archive.glob("*.pt")} == set(schedule)

        def advance(date, ids, batch, encoded, version):
            with torch.no_grad():
                prediction = model.head(encoded).numpy()
            shard = pd.read_csv(native / "A/prediction_shards/A" / f"{date}.csv",
                                dtype={"stock": str}, float_precision="round_trip")
            assert shard.pred_version.eq(version).all()
            native_prediction = align_predictions(test.iloc[ids], shard)
            np.testing.assert_array_equal(native_prediction, base[ids])
            errors.append(float(np.max(np.abs(prediction - native_prediction))))
            if date not in schedule:
                return False
            state = torch.load(archive / f"{date}.pt", map_location="cpu", weights_only=False)
            assert state["version"] == version + 1
            model.load_state_dict(state["model"])
            return True

        features, versions = capture_daily(
            model.adapter, hidden, days, advance, "cpu",
            observe=None if observe is None else lambda date, ids, encoded: observe(date, ids, encoded, model.head))
    else:
        model = new_model().to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        cfg_online = (read_json(native / "run_config.json")["protocol"]
                      if market == "sp500" else read_json(native / "protocol.json"))
        optimizer = online_optimizer(model, cfg_online)
        archive = native / "Adapter_CL"
        groups = make_schedule(test)

        def advance(date, ids, batch, encoded, version):
            prediction = predict_tensor(model, batch)
            journal = read_json(archive / "days" / f"{date}.json")
            assert journal["pred_version"] == version
            shard = archive / "days" / f"{date}.npy"
            assert journal["prediction_sha256"] == sha(shard)
            native_prediction = np.load(shard)
            np.testing.assert_array_equal(native_prediction, base[ids])
            errors.append(float(np.max(np.abs(prediction - native_prediction))))
            selected = groups[date]
            np.testing.assert_array_equal(selected, journal["supervision_row_ids"])
            assert bool(len(selected)) == (date in schedule)
            if not len(selected):
                return False
            rows, horizons = schedule[date]
            np.testing.assert_array_equal(np.sort(rows * 10 + horizons),
                                          (selected[:, None] * 10 + np.arange(10)).ravel())
            targets = read_mature(labels, test, selected, date)
            daily_update(model, optimizer, tensor(hidden[selected], device),
                                 tensor(targets, device), cfg_online["online"]["micro_batch"])
            assert journal["version_after"] == version + 1
            return True

        features, versions = capture_daily(
            model.adapter, hidden, days, advance, device,
            observe=None if observe is None else lambda date, ids, encoded: observe(date, ids, encoded, model.head))
        final = torch.load(archive / "final.pt", map_location="cpu", weights_only=True)
        state_error = max(float((p.cpu() - final[k]).abs().max()) for k, p in model.state_dict().items())
        assert state_error <= 1e-6, state_error
    maximum = max(errors)
    assert maximum <= 1e-7, (market, maximum)
    assert not torch.equal(next(model.adapter.parameters()).cpu(), next(encoder.parameters()))
    return features, dict(baseline_prediction_max_error=maximum, final_state_max_error=state_error,
                          baseline_weight_source="daily checkpoints" if market == "csi300" else "native replay",
                          baseline_device="cpu" if market == "csi300" else str(device)), versions
