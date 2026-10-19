"""Behavioral checks for maturity, rejection, causality and restart invariance."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.kronos_continual import page1c_mars_cl as core
from src.kronos_continual.data import (history_pool, mature_partition, npy_info, raw_summary, PCOLS)
from src.kronos_continual.runner import stream


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def toy():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2025-07-01", periods=110).strftime("%Y-%m-%d").tolist()
    rows = []
    for split, ds in [("train", ["2024-01-02", "2024-01-03"]), ("test", dates[:100])]:
        for d in ds:
            end = "2024-01-17" if split == "train" else dates[dates.index(d)+10]
            for stock in range(6):
                rows.append({"row_id": len(rows), "split": split, "stock": str(stock), "as_of_date": d,
                             "label_start_date": d, "label_end_date": end})
    meta = pd.DataFrame(rows)
    b = rng.normal(0, .02, (len(meta), 10)).astype("float32")
    states = rng.normal(size=(len(meta), 18)).astype("float32")
    y = b + rng.normal(0, .01, b.shape)
    return meta, b, states, y, np.arange(12)


def test_zero_bound_and_rank_gradient():
    adapter = core.ResidualAdapter()
    b, s = torch.randn(9, 10), torch.randn(9, 18)
    assert torch.equal(adapter(b, s), b)
    loss, _ = core.loss_components(adapter(b, s), b, torch.randn(9, 10), np.repeat([1,2,3], 3))
    loss.backward()
    assert adapter.net[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.net[-1].bias.fill_(1e6)
    assert (adapter(b, s)-b).abs().max() <= .0200001
    with pytest.raises(ValueError):
        adapter(b.flatten(), s)


def test_complete_maturity_and_fixed_history():
    m, *_ = toy()
    test = m[m.split.eq("test")]
    d = sorted(test.as_of_date.unique())[39]
    update, shadow = mature_partition(test, d)
    assert shadow.as_of_date.nunique() == 10
    assert not set(update.row_id)&set(shadow.row_id)
    assert (pd.concat([update, shadow]).label_end_date <= d).all()
    # One immature stock removes its entire date from a complete cross-section.
    test = test.copy()
    test.loc[shadow.index[-1], "label_end_date"] = "2099-01-01"
    _, revised = mature_partition(test, d)
    assert shadow.as_of_date.iloc[-1] not in set(revised.as_of_date)
    a = history_pool(m)
    altered = m.copy()
    altered.loc[m.split.eq("test"), "label_end_date"] = "2100-01-01"
    np.testing.assert_array_equal(a, history_pool(altered))


def test_reject_preserves_adam_and_soft_acceptance(monkeypatch):
    m, b, s, y, _ = toy()
    engine = core.MARS(100)
    ids = np.arange(12, 36)
    vals = np.arange(36, 54)
    before = core.state_hash(engine.state_dict())
    monkeypatch.setattr(core, "gate", lambda *_: False)
    log, _, _ = engine.propose(b[ids], s[ids], y[ids], m.iloc[ids].as_of_date.to_numpy(),
                             b[vals], s[vals], y[vals], m.iloc[vals].as_of_date.to_numpy())
    assert not log["accepted"] and before == core.state_hash(engine.state_dict())
    # Exercise the accepted transaction and nonempty Adam state independently
    # of stochastic quality of this synthetic data.
    monkeypatch.setattr(core, "gate", lambda *_: True)
    log, _, _ = engine.propose(b[ids], s[ids], y[ids], m.iloc[ids].as_of_date.to_numpy(),
                             b[vals], s[vals], y[vals], m.iloc[vals].as_of_date.to_numpy())
    assert log["accepted"] and engine.optimizer.state
    saved = engine.state_dict()
    restored = core.MARS(101)
    restored.load_state_dict(saved)
    np.testing.assert_array_equal(engine.predict(b, s), restored.predict(b, s))
    assert core.state_hash(restored.state_dict()) == core.state_hash(saved)


def test_resume_and_unmatured_labels_are_causal(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "gate", lambda *_: True)
    m, b, s, y, history = toy()
    full = core.MARS(100)
    output = tmp_path/"full"
    a = stream(full, m, b, s, y, history, output, 100, "fixture")
    resumed = core.MARS(100)
    partial = tmp_path/"resume"
    assert stream(resumed, m, b, s, y, history, partial, 100, "fixture", stop_after_blocks=3) is None
    restored = core.MARS(102)
    c = stream(restored, m, b, s, y, history, partial, 100, "fixture", resume=True)
    np.testing.assert_array_equal(a[1][PCOLS].to_numpy(), c[1][PCOLS].to_numpy())
    assert core.state_hash(full.state_dict()) == core.state_hash(restored.state_dict())
    check = torch.load(partial/"checkpoints/online_latest.pt", weights_only=False)
    assert check["next_block"] == 5 and len(check["logs"]) == 2
    # Run through the first update while poisoning all labels immature then,
    # including those of already observed inputs and future input states.
    boundary = sorted(m[m.split.eq("test")].as_of_date.unique())[39]
    changed_y, changed_s = y.copy(), s.copy()
    changed_y[m.label_end_date.gt(boundary)] = np.nan
    changed_s[m.as_of_date.gt(boundary)] = 1e6
    causal = core.MARS(100)
    altered = tmp_path/"future_perturbed"
    stream(causal, m, b, changed_s, changed_y, history, altered, 100, "fixture", stop_after_blocks=2)
    expected = torch.load(output/"checkpoints/block_001.pt", weights_only=False)
    observed = torch.load(altered/"checkpoints/block_001.pt", weights_only=False)
    assert core.state_hash(expected["engine"]) == core.state_hash(observed["engine"])
    for p in sorted((altered/"prediction_days").glob("*.csv")):
        assert p.read_bytes() == (output/"prediction_days"/p.name).read_bytes()


def test_payload_truncation_and_raw_state(tmp_path):
    path = tmp_path/"hidden.npy"
    np.save(path, np.ones((5, 512), np.float32))
    assert npy_info(path, (5,512))["valid"]
    with path.open("r+b") as f:
        f.truncate(path.stat().st_size-100)
    assert not npy_info(path, (5,512))["valid"]
    window = np.ones((90,6))
    result = raw_summary(window)
    np.testing.assert_array_equal(result, np.r_[np.ones(6),np.zeros(6),np.ones(6)])
    window[-1, 3] = np.nan
    with pytest.raises(ValueError):
        raw_summary(window)


def test_gate_constraints_and_nonfinite():
    current = {"path_rankic": .05, "path_mse": .01}
    assert core.gate(current, {"path_rankic": .051, "path_mse": .01})
    assert not core.gate(current, {"path_rankic": .047, "path_mse": .009})
    assert not core.gate(current, {"path_rankic": .06, "path_mse": .011})
    assert not core.gate(current, current)
    with pytest.raises(FloatingPointError):
        core.gate(current, {"path_rankic": np.nan, "path_mse": .01})
