"""Frozen Page1-C and maturity-constrained residual adaptation."""
from __future__ import annotations
import copy
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, d=512, bottleneck=64, dropout=.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d)

    def forward(self, h):
        return h + self.up(self.drop(F.gelu(self.down(self.ln(h)))))


class Page1CModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.head = nn.Linear(512, 10)

    def forward(self, h):
        return self.head(self.adapter(h))


class ResidualAdapter(nn.Module):
    delta_max = .02

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28, 32), nn.GELU(), nn.Linear(32, 10))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, state):
        if base.ndim != 2 or base.shape[1] != 10 or state.shape != (len(base), 18):
            raise ValueError("expected base [N,10] and state [N,18]")
        return base + self.delta_max * torch.tanh(self.net(torch.cat([base, state], -1)))


def finite(array, name):
    ok = bool(torch.isfinite(array).all()) if isinstance(array, torch.Tensor) else bool(np.isfinite(np.asarray(array)).all())
    if not ok:
        raise FloatingPointError(f"nonfinite {name}; fixed configuration will not be retried")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def state_hash(value):
    """Stable nested hash of model and optimizer tensors and scalar settings."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        elif isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode())
            h.update(a.tobytes())
        else:
            h.update(pickle.dumps(x, protocol=4))
    visit(value)
    return h.hexdigest()


def pairwise_rank_loss(pred, target, max_pairs=4096):
    """One date only; deterministic unordered pairs; target ties excluded."""
    ps, ys = pred.mean(1), target.mean(1)
    if len(ps) < 2:
        return ps.sum() * 0
    ii, jj = torch.triu_indices(len(ps), len(ps), offset=1, device=pred.device)
    if len(ii) > max_pairs:
        keep = torch.linspace(0, len(ii)-1, max_pairs, device=pred.device).long()
        ii, jj = ii[keep], jj[keep]
    diff = ys[ii] - ys[jj]
    mask = diff != 0
    if not bool(mask.any()):
        return ps.sum() * 0
    return F.softplus(-torch.sign(diff[mask]) * (ps[ii[mask]]-ps[jj[mask]])).mean()


def loss_components(pred, base, target, dates):
    path = F.smooth_l1_loss(pred, target)
    dates = np.asarray(dates)
    ranks = [pairwise_rank_loss(pred[ix], target[ix]) for d in np.unique(dates)
             if len(ix := np.flatnonzero(dates == d)) > 1]
    rank = torch.stack(ranks).mean() if ranks else pred.sum() * 0
    anchor = F.mse_loss(pred, base)
    total = .3 * path + .5 * rank + .2 * anchor
    finite(total, "loss")
    return total, {"path": float(path.detach()), "rank": float(rank.detach()),
                   "anchor": float(anchor.detach()), "total": float(total.detach())}


def shadow_metrics(pred, base, target, dates):
    finite(pred, "shadow predictions")
    finite(target, "shadow labels")
    daily = []
    for date in sorted(set(dates)):
        ix = np.flatnonzero(np.asarray(dates) == date)
        p, b, y = pred[ix], base[ix], target[ix]
        daily.append({"as_of_date": date, "rows": len(ix),
                      "path_rankic": pd.Series(p.mean(1)).corr(pd.Series(y.mean(1)), method="spearman"),
                      "endpoint_rankic": pd.Series(p[:, -1]).corr(pd.Series(y[:, -1]), method="spearman"),
                      "ranking_vs_base": pd.Series(p.mean(1)).corr(pd.Series(b.mean(1)), method="spearman"),
                      "path_mse": float(np.square(p-y).mean()),
                      "residual_abs_mean": float(np.abs(p-b).mean())})
    frame = pd.DataFrame(daily)
    finite(frame.drop(columns="as_of_date").to_numpy(), "shadow metrics")
    return {"path_rankic": float(frame.path_rankic.mean()), "endpoint_rankic": float(frame.endpoint_rankic.mean()),
            "path_mse": float(np.square(pred-target).mean()), "residual_abs_mean": float(np.abs(pred-base).mean()),
            "residual_abs_max": float(np.abs(pred-base).max())}, daily


def gate(current, candidate):
    finite(list(current.values()) + list(candidate.values()), "gate metrics")
    return bool(candidate["path_rankic"] >= current["path_rankic"]-.002
                and candidate["path_mse"] <= 1.05*current["path_mse"]
                and (candidate["path_rankic"] > current["path_rankic"] or candidate["path_mse"] < current["path_mse"]))


class MARS:
    """Owns only residual parameters and their AdamW state."""
    def __init__(self, seed, device="cpu"):
        seed_all(seed)
        self.device = torch.device(device)
        self.adapter = ResidualAdapter().to(device).eval()
        self.optimizer = self._optimizer(self.adapter)

    @staticmethod
    def _optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def state_dict(self):
        return copy.deepcopy({"model": self.adapter.state_dict(), "optimizer": self.optimizer.state_dict()})

    def load_state_dict(self, state):
        self.adapter.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.adapter.eval()

    @torch.no_grad()
    def predict(self, base, state, model=None):
        model = self.adapter if model is None else model
        model.eval()
        result = []
        for start in range(0, len(base), 512):
            b = torch.tensor(np.asarray(base[start:start+512]), dtype=torch.float32, device=self.device)
            s = torch.tensor(np.asarray(state[start:start+512]), dtype=torch.float32, device=self.device)
            p = model(b, s)
            finite(p, "online prediction")
            if bool((p-b).abs().gt(.02000001).any()):
                raise AssertionError("residual bound exceeded")
            result.append(p.cpu().numpy())
        return np.concatenate(result)

    def propose(self, base, states, targets, dates, val_base, val_states, val_targets, val_dates):
        """Candidate copies Adam state. Reject is a no-op; soft write is re-gated.

        Acceptance retains candidate Adam moments without interpolating moments;
        only parameters receive the fixed 0.2 interpolation.
        """
        before = state_hash(self.state_dict())
        candidate = copy.deepcopy(self.adapter)
        opt = self._optimizer(candidate)
        opt.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        groups = [np.flatnonzero(np.asarray(dates) == d) for d in sorted(set(dates))]
        history = []
        for epoch in range(2):
            candidate.train()
            for g in np.random.permutation(len(groups)):
                ix = groups[g]
                b = torch.tensor(np.asarray(base[ix]), dtype=torch.float32, device=self.device)
                s = torch.tensor(np.asarray(states[ix]), dtype=torch.float32, device=self.device)
                y = torch.tensor(np.asarray(targets[ix]), dtype=torch.float32, device=self.device)
                loss, parts = loss_components(candidate(b, s), b, y, np.asarray(dates)[ix])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in candidate.parameters():
                    if parameter.grad is not None:
                        finite(parameter.grad, "candidate gradient")
                nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
                opt.step()
                history.append({"epoch": epoch+1, "date": str(np.asarray(dates)[ix[0]]), "rows": len(ix), **parts})
        for p in candidate.parameters():
            finite(p, "candidate parameter")
        current_pred = self.predict(val_base, val_states)
        cp = self.predict(val_base, val_states, candidate)
        cm, cd = shadow_metrics(current_pred, val_base, val_targets, val_dates)
        pm, pdaily = shadow_metrics(cp, val_base, val_targets, val_dates)
        accepted = gate(cm, pm)
        decision = "rejected_candidate_gate"
        soft = copy.deepcopy(self.adapter)
        soft_metrics, soft_daily = {}, []
        if accepted:
            with torch.no_grad():
                for dst, old, new in zip(soft.parameters(), self.adapter.parameters(), candidate.parameters()):
                    dst.copy_(.8*old + .2*new)
            sp = self.predict(val_base, val_states, soft)
            soft_metrics, soft_daily = shadow_metrics(sp, val_base, val_targets, val_dates)
            accepted = gate(cm, soft_metrics)
            decision = "accepted" if accepted else "rejected_soft_gate"
            if accepted:
                self.adapter.load_state_dict(soft.state_dict())
                self.optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
        if not accepted and before != state_hash(self.state_dict()):
            raise AssertionError("rejected proposal mutated current model/optimizer")
        after = state_hash(self.state_dict())
        log = {"decision": decision, "accepted": accepted, "before_hash": before, "after_hash": after,
               **{f"current_{k}": v for k, v in cm.items()}, **{f"candidate_{k}": v for k, v in pm.items()},
               **{f"soft_{k}": v for k, v in soft_metrics.items()}}
        daily = [{**r, "model": name} for name, rows in [("current", cd), ("candidate", pdaily), ("soft", soft_daily)] for r in rows]
        return log, daily, history
