"""Causal common/relative Prediction calibration, with an immutable online A stream.

Run: python scripts/experiments/run_prediction_split_residual.py run
Report only: python scripts/experiments/run_prediction_split_residual.py report
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.experiments import run_hca_gain_attribution as engine
from evaluate.metrics import EvaluationConfig, clean_json, evaluate_frame
from src.doubleadapt.compare import indices, return_stats

DATA = ROOT / 'experiments/csi300_aligned'
HISTORY = DATA / 'models/prediction_dct_full_20261102'
SOURCE = HISTORY / 'Prediction_Live_A'
DEFAULT_OUT = DATA / 'models/prediction_split_residual_20261103'
GRID = np.array([0., .01, .03, .1, .3, 1.], dtype=np.float64)
KEYS, PC, AC = engine.KEYS, engine.PCOLS, engine.ACOLS
LABELS = dict(A='在线 A（无残差）', Common='仅共同修正', Individual='仅个股修正',
              Dual='共同＋个股修正', Unsplit='未分离独立校准头', PCA='原 HCA-PCA', DCT='原 HCA-DCT')


def save(path, value):
    engine.old.atomic_json(Path(path), clean_json(value))


def npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **values)
    path.with_suffix('.tmp').replace(path)


def initial_state():
    return dict(common=np.zeros((6, 2)), individual=np.zeros((6, 2, 512)),
                unsplit=np.zeros((6, 2, 513)))


def corrections(state, x, basis):
    common = state['common'] @ basis.T
    centered = x - x.mean(0, keepdims=True)
    individual = np.einsum('nd,grd,kr->gnk', centered, state['individual'], basis)
    augmented = np.column_stack([x, np.ones(len(x))])
    unsplit = np.einsum('nd,grd,kr->gnk', augmented, state['unsplit'], basis)
    return common, individual, unsplit


def step(state, xs, residual, basis, norms):
    """xs[k] and residual[k] contain exactly the original 32 labels for horizon k.

    Loss = 1/(2*10) sum_k mean_i (prediction_ik - target_ik)^2.
    Horizon-specific centering is essential: support sets can differ across k.
    """
    out = {key: value.copy() for key, value in state.items()}
    mean = np.array([v.mean() for v in residual])
    coeff = mean @ basis
    out['common'] += GRID[:, None] * (coeff - state['common'])
    gi = np.zeros_like(state['individual'])
    gu = np.zeros_like(state['unsplit'])
    for k, (x, r) in enumerate(zip(xs, residual)):
        xc = x - x.mean(0, keepdims=True)
        xa = np.column_stack([x, np.ones(len(x))])
        pi = np.einsum('nd,grd,r->gn', xc, state['individual'], basis[k])
        pu = np.einsum('nd,grd,r->gn', xa, state['unsplit'], basis[k])
        gi += np.einsum('gn,nd,r->grd', pi - (r-r.mean())[None, :], xc, basis[k]) / (len(x)*10)
        gu += np.einsum('gn,nd,r->grd', pu - r[None, :], xa, basis[k]) / (len(x)*10)
    out['individual'] -= (GRID*10/norms[0])[:, None, None] * gi
    out['unsplit'] -= (GRID*10/norms[1])[:, None, None] * gu
    assert all(np.isfinite(v).all() for v in out.values())
    return out


def validate_kernel():
    """Compare actual numpy updates against autograd, including differing sets."""
    rng = np.random.default_rng(735)
    b, _ = np.linalg.qr(rng.normal(size=(10, 2)))
    xs = [rng.normal(size=(32, 512)) for _ in range(10)]
    rs = [rng.normal(scale=.03, size=32) for _ in range(10)]
    state = initial_state()
    for k in state:
        state[k] = rng.normal(scale=.0001, size=state[k].shape)
        state[k][0] = 0
    norms = (490., 513.)
    changed = step(state, xs, rs, b, norms)
    for key, centered, divisor in [('individual', True, norms[0]), ('unsplit', False, norms[1])]:
        w = torch.tensor(state[key][3], requires_grad=True, dtype=torch.float64)
        loss = 0
        for k, (x, r) in enumerate(zip(xs, rs)):
            if centered:
                x, r = x-x.mean(0), r-r.mean()
            else:
                x = np.column_stack([x, np.ones(len(x))])
            pred = torch.tensor(x) @ w.T @ torch.tensor(b[k])
            loss = loss + .5*((pred-torch.tensor(r))**2).mean()/10
        loss.backward()
        expected = w.detach().numpy()-GRID[3]*10/divisor*w.grad.numpy()
        np.testing.assert_allclose(changed[key][3], expected, atol=1e-16, rtol=1e-12)
    for key in state:
        np.testing.assert_array_equal(changed[key][0], state[key][0])
    return dict(numpy_matches_torch_autograd=True, per_horizon_supports_tested=True, zero_lr_identity=True)


def run(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'runner_snapshot.py')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = dict(status='locked_before_efficacy', checkpoint='Prediction', exploratory=True,
        grid=GRID.tolist(), rank=2, basis='existing training-only PCA; unchanged',
        base='replay immutable historical A checkpoints; no new base training',
        common='EMA of PCA-projected current-base residual support means',
        individual='fixed non-affine LN(eps=1e-5) frozen h; centered support x/residual; bias-free 512x2',
        unsplit='same LN h, bias, full residual, independent SGD 513x2',
        loss='half squared error, equal horizons, exact per-horizon original 32-stock feedback',
        scaling='individual: mean ||x-mean_support(x)||^2; unsplit: mean ||[x,1]||^2',
        norm_timing='causal prefix input-only mean before day80; frozen from all first80 support inputs at day80',
        normalization_note='early prefix norms vary causally to avoid future-input use; selection/evaluation norms fixed',
        selection='date indices80:140 whose full labels end strictly before date140; independent component MSE',
        evaluation='indices140:226; no retuning; state continuous through embargo',
        update_order='save every candidate -> read mature feedback -> compute residual with PRE-update A -> update calibrators -> load historical POST-update A',
        feedback='216 days x 32 stocks x 10 horizons = 69120 unique scalars, shared among candidates',
        base_label_reuse='same authorized mature labels can supervise A and calibration; no new unique supervision',
        bootstrap=dict(block=10,repetitions=5000,seed=20261013),
        limitations=['previously inspected window; no independent confirmation', 'single market and fixed panel',
                    'support common mean may be a noisy/biased estimate of full-panel mean',
                    'calibration state may become stale as base A updates; no lifecycle mechanism in this test'])
    save(out/'locked_config.json', config)
    checks = validate_kernel()
    bundle = engine.load_inputs_for_checkpoint(engine.full.MODELS['Prediction'])
    meta, test, hidden, values, dates, stocks, rows, hh, _, schedule, releases, hidden_hash, model = bundle
    basis = np.load(HISTORY/'PCA_basis.npy').astype(np.float64)
    np.testing.assert_allclose(basis.T@basis, np.eye(2), atol=3e-7)
    # No future labels are loaded into the training loop except through ReleasedLabels.
    guard = engine.pilot.ReleasedLabels(values, releases)
    source = pd.read_csv(SOURCE/'causal_predictions.csv', dtype={'stock':str}).sort_values(KEYS).reset_index(drop=True)
    access = pd.read_csv(SOURCE/'access_log.csv', dtype={'stock':str})
    expected_schedule = pd.DataFrame([dict(update_date=d, **e) for d in dates for e in schedule[d]])
    cols = ['update_date','row_id','stock','origin_date','horizon','label_date','complete_date']
    pd.testing.assert_frame_equal(access[cols].reset_index(drop=True), expected_schedule[cols].reset_index(drop=True))
    assert len(access)==69120 and not access.duplicated(['row_id','horizon']).any()
    x = {d:F.layer_norm(hh[d], (512,), eps=1e-5).numpy().astype(np.float64) for d in dates}
    lookups = {d:{int(r):j for j,r in enumerate(rows[d].row_id)} for d in dates}
    support = {}
    norm_by_day = []
    for d in dates:
        day = []
        for k in range(1,11):
            events = [e for e in schedule[d] if e['horizon']==k]
            if events:
                assert len(events)==32 and len({e['origin_date'] for e in events})==1
                origin = events[0]['origin_date']
                take = np.array([lookups[origin][e['row_id']] for e in events])
                day.append((origin,take,events))
        support[d] = day
        if day:
            ni=[];nu=[]
            for origin,take,_ in day:
                xx=x[origin][take]
                ni.append(np.mean(np.sum((xx-xx.mean(0))**2,axis=1)))
                nu.append(np.mean(np.sum(xx**2,axis=1))+1)
            norm_by_day.append((d,float(np.mean(ni)),float(np.mean(nu))))
    norm_frame=pd.DataFrame(norm_by_day,columns=['date','individual','unsplit'])
    frozen=norm_frame[norm_frame.date<dates[80]][['individual','unsplit']].mean().to_numpy()
    np.testing.assert_array_less(0, frozen)
    windows=pd.read_csv(DATA/'reference/date_windows.csv',dtype=str).drop_duplicates('as_of_date').set_index('as_of_date')
    select=[i for i in range(80,140) if windows.loc[dates[i],'label_end_date']<dates[140]]
    assert len(select)==50
    save(out/'splits.json',dict(dates=dates,selection_indices=select,evaluation_indices=list(range(140,226)),
        selection_last_label=windows.loc[dates[select[-1]],'label_end_date'],
        frozen_norm_individual=frozen[0],frozen_norm_unsplit=frozen[1]))
    norm_frame.to_csv(out/'support_feature_norms.csv',index=False)
    manifests={'initial_checkpoint':engine.full.MODELS['Prediction']['checkpoint'],
        'A_predictions':SOURCE/'causal_predictions.csv','A_access_log':SOURCE/'access_log.csv',
        'PCA_basis':HISTORY/'PCA_basis.npy','hidden':DATA/'reference/hidden_cache/hidden_test.npy',
        'labels':DATA/'doubleadapt_cache_v2/y.npy','metadata':DATA/'doubleadapt_cache_v2/meta.csv'}
    save(out/'inputs.json',{key:dict(path=str(path),sha256=engine.old.sha256(path)) for key,path in manifests.items()})
    # Keep source checkpoint hashes; source files must never be modified.
    cps=sorted((SOURCE/'checkpoints/A').glob('*.pt'))
    cp_hashes={p.name:engine.old.sha256(p) for p in cps}
    save(out/'source_checkpoint_hashes.json',cp_hashes)
    assert len(cps)==216
    state=initial_state();consumed=set();journals=[];base_error=0.;zero_mean=0.;started=time.perf_counter()
    for pos,d in enumerate(dates):
        model.eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            base=model(hh[d]).numpy().astype(np.float64)
        archived=source[source.as_of_date.eq(d)].reset_index(drop=True)
        pd.testing.assert_frame_equal(archived[KEYS],rows[d][KEYS].reset_index(drop=True))
        err=float(abs(base-archived[PC].to_numpy()).max());base_error=max(base_error,err)
        assert err<=1e-7
        base=archived[PC].to_numpy()  # Exact archived baseline in all output comparisons.
        common,individual,unsplit=corrections(state,x[d],basis)
        zero_mean=max(zero_mean,float(abs(individual.mean(axis=1)).max()))
        np.testing.assert_allclose(individual.mean(axis=1),0,atol=1e-13)
        assert np.isfinite(common).all() and np.isfinite(individual).all() and np.isfinite(unsplit).all()
        for corr in [common,individual,unsplit]:np.testing.assert_array_equal(corr[0],np.zeros_like(corr[0]))
        shard=out/'candidate_shards'/f'{d}.npz'
        # These are every candidate's complete outputs in additive form, before any label read.
        npz(shard,base=base,common=common,individual=individual,unsplit=unsplit,
            row_id=rows[d].row_id.to_numpy(),stock=rows[d].stock.to_numpy(dtype='U'))
        digest=engine.old.sha256(shard)
        if support[d]:
            xx=[];rr=[];reads=[]
            for k,(origin,take,events) in enumerate(support[d]):
                targets=[]
                for e in events:
                    key=(e['row_id'],e['horizon']);assert key not in consumed
                    targets.append(guard.read(e['row_id'],k,d));consumed.add(key)
                    reads.append(dict(update_date=d,**e,prediction_sha256=digest))
                with torch.no_grad():
                    pred=model(hh[origin][take]).numpy()[:,k].astype(np.float64)
                xx.append(x[origin][take]);rr.append(np.array(targets)-pred)
            norms=frozen if pos>=80 else norm_frame[norm_frame.date<=d][['individual','unsplit']].mean().to_numpy()
            norm_digest=hashlib.sha256(np.asarray(norms).tobytes()).hexdigest()
            # Save a real interruption checkpoint, then reload it before proceeding.
            if pos==139:
                npz(out/'resume_state.npz',**state)
                saved=np.load(out/'resume_state.npz');resumed={k:saved[k].copy() for k in state}
                expected=step(state,xx,rr,basis,norms)
                state=step(resumed,xx,rr,basis,norms)
                for key in state:np.testing.assert_array_equal(state[key],expected[key])
                checks['resume_file_load_update_exact']=True
            else:
                state=step(state,xx,rr,basis,norms)
            # Simulate a future-label poison: current and past authorized reads unchanged.
            cutoff=d
            class PoisonFuture:
                def read(self, row_id, k, today):
                    if releases[row_id][-1]>cutoff:return 1e8
                    return guard.read(row_id,k,today)
            poison=PoisonFuture()
            for k,(_,_,events) in enumerate(support[d]):
                for e in events:assert poison.read(e['row_id'],k,d)==guard.read(e['row_id'],k,d)
            npz(out/'update_inputs'/f'{d}.npz',x=np.stack(xx),residual=np.stack(rr),norms=norms)
            engine.old.atomic_csv(out/'access_shards'/f'{d}.csv',pd.DataFrame(reads))
            journals.extend(reads)
            npz(out/'state_shards'/f'{d}.npz',**state)
            save(out/'update_journals'/f'{d}.json',dict(date=d,base_pre_update_max_error=err,
                norms=norms.tolist(),norm_digest=norm_digest,prediction_sha256=digest,
                common_state_norm=np.linalg.norm(state['common'],axis=1).tolist(),
                individual_weight_norm=np.linalg.norm(state['individual'],axis=(1,2)).tolist(),
                unsplit_weight_norm=np.linalg.norm(state['unsplit'],axis=(1,2)).tolist()))
        for key,v in model.state_dict().items():assert torch.equal(v,before[key])
        # A is updated ONLY by loading its historical post-update checkpoint.
        cp=SOURCE/'checkpoints/A'/f'{d}.pt'
        if cp.exists():model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
        if pos%40==0:print(f'CALIBRATION {pos+1}/226 {d}',flush=True)
    engine.old.atomic_csv(out/'access_log.csv',pd.DataFrame(journals))
    assert len(consumed)==69120
    pd.testing.assert_frame_equal(pd.DataFrame(journals)[cols].reset_index(drop=True),access[cols].reset_index(drop=True))
    for p in cps:assert engine.old.sha256(p)==cp_hashes[p.name]
    checks.update(base_forecast_max_abs_error=base_error,individual_max_abs_mean=zero_mean,
        immutable_A=True,source_checkpoint_hashes_unchanged=True,same_feedback_keys=True,
        labels=69120,updates=216,zero_candidates_exact=True,prediction_written_before_labels=True)
    # Full 226-day state replay, plus resume replay of entire suffix.
    replay=initial_state();suffix=None
    for pos,d in enumerate(dates):
        saved=np.load(out/'candidate_shards'/f'{d}.npz')
        c,i,u=corrections(replay,x[d],basis)
        for key,v in [('common',c),('individual',i),('unsplit',u)]:np.testing.assert_array_equal(v,saved[key])
        if pos==139:
            with np.load(out/'resume_state.npz') as q:suffix={key:q[key].copy() for key in replay}
        update=out/'update_inputs'/f'{d}.npz'
        if update.exists():
            with np.load(update) as q:
                replay=step(replay,list(q['x']),list(q['residual']),basis,q['norms'])
                if suffix is not None:
                    suffix=step(suffix,list(q['x']),list(q['residual']),basis,q['norms'])
                    for key in replay:np.testing.assert_array_equal(replay[key],suffix[key])
            with np.load(out/'state_shards'/f'{d}.npz') as q:
                for key in replay:np.testing.assert_array_equal(replay[key],q[key])
    checks.update(full_state_replay_exact=True,resumed_suffix_exact=True)
    save(out/'training_checks.json',checks)
    save(out/'training_complete.json',dict(status='passed',seconds=time.perf_counter()-started))
    print('CALIBRATION COMPLETE '+str(checks),flush=True)


def loss_summary(pred, actual):
    # Identical forecasts must have identical reduction order regardless of pandas layout.
    e=np.ascontiguousarray(actual)-np.ascontiguousarray(pred)
    common=e.mean(axis=0,keepdims=True)
    mse=float(np.mean(e**2));cm=float(np.mean(common**2));rel=float(np.mean((e-common)**2))
    np.testing.assert_allclose(mse,cm+rel,atol=1e-15,rtol=1e-12)
    return dict(mse=mse,mae=float(np.mean(abs(e))),common_mse=cm,relative_mse=rel,
        path_direction=float(np.mean((pred.mean(1)>0)==(actual.mean(1)>0))),
        endpoint_direction=float(np.mean((pred[:,-1]>0)==(actual[:,-1]>0))),n=len(e))


def pick(scores):
    values=np.asarray(scores)
    minimum=values.min()
    return int(np.flatnonzero(values<=minimum+max(1e-18,abs(minimum)*1e-12))[0])


def verify_future_poison(out):
    """Replay real prefixes against label stores with all unavailable paths poisoned.

    Historical A is an immutable external input here. This verifies the new heads'
    feedback/state trajectory; it does not pretend to re-audit training of A.
    """
    split=json.loads((out/'splits.json').read_text(encoding='utf8'));dates=split['dates']
    accesses=pd.read_csv(out/'access_log.csv',dtype={'stock':str})
    meta=pd.read_csv(DATA/'doubleadapt_cache_v2/meta.csv',usecols=['row_id','label_end_date'])
    ends=meta.set_index('row_id').label_end_date.to_dict()
    windows=pd.read_csv(DATA/'reference/date_windows.csv',dtype=str).drop_duplicates('as_of_date').set_index('as_of_date')
    releases={int(r.row_id):str(windows.loc[r.origin_date,'label_dates']).split('|')
        for r in accesses[['row_id','origin_date']].drop_duplicates().itertuples(index=False)}
    values=np.load(DATA/'doubleadapt_cache_v2/y.npy',mmap_mode='r')
    basis=np.load(HISTORY/'PCA_basis.npy').astype(np.float64)
    groups={d:g for d,g in accesses.groupby('update_date',sort=False)}
    results=[]
    for endpos in [80,140,190,226]:
        cutoff=dates[endpos-1]
        class PoisonedValues:
            count=0
            def __getitem__(self,key):
                row,k=key
                if ends[int(row)]>cutoff:
                    self.count+=1
                    return float(values[row,k])+1e6
                return values[row,k]
        poison=PoisonedValues()
        changed=engine.pilot.ReleasedLabels(poison,releases)
        clean=engine.pilot.ReleasedLabels(values,releases)
        state=initial_state();seen=[]
        # Establish that the alternate store actually changes unavailable values,
        # and the production maturity guard prevents accessing such paths.
        future=[r for r in releases if ends[r]>cutoff]
        if future:
            row=future[0]
            assert poison[row,0]!=values[row,0]
            try:
                changed.read(row,0,cutoff)
            except AssertionError:
                pass
            else:
                raise AssertionError('future complete path accepted')
        for d in dates[:endpos]:
            if d not in groups:continue
            with np.load(out/'update_inputs'/f'{d}.npz') as q:
                rr=[]
                for k in range(10):
                    events=groups[d][groups[d].horizon.eq(k+1)]
                    delta=[]
                    for e in events.itertuples(index=False):
                        delta.append(changed.read(e.row_id,k,d)-clean.read(e.row_id,k,d))
                        seen.append((int(e.row_id),k+1))
                    rr.append(q['residual'][k]+np.asarray(delta))
                state=step(state,list(q['x']),rr,basis,q['norms'])
            with np.load(out/'state_shards'/f'{d}.npz') as expected:
                for key in state:np.testing.assert_array_equal(state[key],expected[key])
        expected=accesses[accesses.update_date.le(cutoff)]
        assert seen==list(expected[['row_id','horizon']].itertuples(index=False,name=None))
        results.append(dict(cutoff=cutoff,days=endpos,state_and_access_trajectory_exact=True,
            changed_future_value_demonstrated=bool(future),future_guard_rejected=bool(future)))
    save(out/'future_poison_checks.json',dict(status='passed',prefixes=results,
        note='Real guarded label reads rerun; future paths shifted +1e6. Matches full state trajectory and access keys. Base A remains the saved external input.'))


def report(out):
    assert (out/'training_complete.json').exists()
    verify_future_poison(out)
    split=json.loads((out/'splits.json').read_text(encoding='utf8'))
    dates=split['dates'];selection=set(split['selection_indices']);evaluation=dates[140:]
    ref=pd.read_csv(SOURCE/'predictions.csv',dtype={'stock':str}).sort_values(KEYS).reset_index(drop=True)
    refs={d:f.reset_index(drop=True) for d,f in ref.groupby('as_of_date',sort=True)}
    # Selection scoring is deliberately restricted before any final-period scoring.
    score=np.zeros((3,6));count=0
    for pos in sorted(selection):
        d=dates[pos];q=np.load(out/'candidate_shards'/f'{d}.npz');y=refs[d][AC].to_numpy();base=q['base']
        for g in range(6):
            score[0,g]+=loss_summary(base+q['common'][g],y)['common_mse']
            score[1,g]+=loss_summary(base+q['individual'][g],y)['relative_mse']
            score[2,g]+=loss_summary(base+q['unsplit'][g],y)['mse']
        count+=1
    score/=count
    chosen={name:pick(score[j]) for j,name in enumerate(['common','individual','unsplit'])}
    selection_rows=[dict(head=name,alpha=float(GRID[g]),selection_loss=score[j,g],selected=g==chosen[name])
        for j,name in enumerate(chosen) for g in range(6)]
    pd.DataFrame(selection_rows).to_csv(out/'selection_scores.csv',index=False)
    save(out/'selection.json',dict(indices=chosen,alphas={k:GRID[v] for k,v in chosen.items()},
        rule='independent per-component loss; no dual/output/portfolio tuning',dates=count))
    # Report every prespecified candidate, without using final-period losses to select.
    grid_rows=[];rank_checks=0
    for pos,d in enumerate(dates):
        q=np.load(out/'candidate_shards'/f'{d}.npz');y=refs[d][AC].to_numpy();base=q['base']
        baseline=loss_summary(base,y)
        for g,alpha in enumerate(GRID):
            shifted=base+q['common'][g]
            for left,right in [(base.mean(1),shifted.mean(1)),(base[:,-1],shifted[:,-1])]:
                np.testing.assert_array_equal(np.argsort(left,kind='stable'),np.argsort(right,kind='stable'))
                rank_checks+=1
            for name,correction in [('Common',q['common'][g]),('Individual',q['individual'][g]),
                                    ('Dual_same_alpha',q['common'][g]+q['individual'][g]),('Unsplit',q['unsplit'][g])]:
                stats=loss_summary(base+correction,y)
                grid_rows.append(dict(method=name,alpha=alpha,date=d,is_selection=pos in selection,
                    is_evaluation=pos>=140,mse_delta=stats['mse']-baseline['mse'],
                    common_delta=stats['common_mse']-baseline['common_mse'],
                    relative_delta=stats['relative_mse']-baseline['relative_mse'],
                    mae_delta=stats['mae']-baseline['mae'],**stats))
    grid_daily=pd.DataFrame(grid_rows)
    grid_daily.to_csv(out/'all_candidate_daily_metrics.csv',index=False)
    grid_table=[]
    for window,mask in [('selection_50',grid_daily.is_selection),('evaluation_86',grid_daily.is_evaluation)]:
        for (name,alpha),g in grid_daily[mask].groupby(['method','alpha']):
            baseline_mse=(g.mse-g.mse_delta).mean()
            grid_table.append(dict(window=window,method=name,alpha=alpha,mse=g.mse.mean(),
                mse_relative_improvement=1-g.mse.mean()/baseline_mse,mae=g.mae.mean(),
                mse_delta=g.mse_delta.mean(),common_delta=g.common_delta.mean(),relative_delta=g.relative_delta.mean()))
    pd.DataFrame(grid_table).to_csv(out/'all_candidate_metrics.csv',index=False)
    frames={name:[] for name in ['A','Common','Individual','Dual','Unsplit']};daily=[];horizons=[];branch=[]
    for pos,d in enumerate(dates):
        f=refs[d];y=f[AC].to_numpy();q=np.load(out/'candidate_shards'/f'{d}.npz')
        base=q['base'];a=np.broadcast_to(q['common'][chosen['common']],base.shape)
        v=q['individual'][chosen['individual']];u=q['unsplit'][chosen['unsplit']]
        preds=dict(A=base,Common=base+a,Individual=base+v,Dual=base+a+v,Unsplit=base+u)
        np.testing.assert_array_equal(preds['Dual'],preds['A']+a+v)
        for name,pred in preds.items():
            g=f.copy();g[PC]=pred;g['method']=name
            for prefix,cs in [('pred',PC),('actual',AC)]:
                g[prefix+'_signal']=g[cs].mean(1);g[prefix+'_path_mean_return']=g[cs].mean(1);g[prefix+'_endpoint_return']=g[cs[-1]]
            frames[name].append(g)
            stats=loss_summary(pred,y);daily.append(dict(method=name,date=d,**stats))
            for k in range(10):
                er=y[:,k]-pred[:,k]
                horizons.append(dict(method=name,date=d,horizon=k+1,mse=float(np.mean(er**2)),
                    mae=float(np.mean(abs(er))),n=len(er),direction=float(np.mean((pred[:,k]>0)==(y[:,k]>0)))))
        bb=f[KEYS].copy()
        for prefix,arr in [('common',a),('individual',v),('unsplit',u)]:
            for k in range(10):bb[f'{prefix}_{k+1}']=arr[:,k]
        branch.append(bb)
    pd.concat(branch,ignore_index=True).to_csv(out/'selected_branch_outputs.csv',index=False)
    for name in frames:frames[name]=pd.concat(frames[name],ignore_index=True)
    for name in ['PCA','DCT']:
        frames[name]=pd.read_csv(HISTORY/('Prediction_Live_'+name)/'predictions.csv',dtype={'stock':str}).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(frames[name][KEYS+AC],ref[KEYS+AC])
        for d,g in frames[name].groupby('as_of_date',sort=True):
            pred=g[PC].to_numpy();y=g[AC].to_numpy()
            daily.append(dict(method=name,date=d,**loss_summary(pred,y)))
            for k in range(10):
                er=y[:,k]-pred[:,k]
                horizons.append(dict(method=name,date=d,horizon=k+1,mse=float(np.mean(er**2)),mae=float(np.mean(abs(er))),
                    n=len(er),direction=float(np.mean((pred[:,k]>0)==(y[:,k]>0)))))
    daily=pd.DataFrame(daily).sort_values(['method','date']).reset_index(drop=True)
    hd=pd.DataFrame(horizons);daily.to_csv(out/'daily_prediction_metrics.csv',index=False)
    hd.to_csv(out/'daily_horizon_metrics.csv',index=False)
    cfg=EvaluationConfig(top_k=30,drop_n=3,min_holding_days=5,cost_rate=.0015,annualization_days=252,bootstrap_reps=0)
    metrics=[];by_horizon=[];portfolios={};evaluations={}
    for name,f in frames.items():
        p=out/name;p.mkdir(exist_ok=True)
        f.to_csv(p/'predictions.csv',index=False)
        for window,ds in [('evaluation_86',evaluation),('full_226',dates)]:
            subset=f[f.as_of_date.isin(ds)].copy()
            result=evaluate_frame(subset,cfg);save(p/(window+'.json'),result);evaluations[name,window]=result
            dd=daily[(daily.method==name)&daily.date.isin(ds)]
            pooled=result['metrics']['trajectory_pooled']
            for mode in ['path','endpoint']:
                m=result['metrics'][mode];port=pd.DataFrame(result['daily'][mode]['portfolio'])
                cross=pd.DataFrame(result['daily'][mode]['cross_section'])
                q=port.merge(cross,on='as_of_date',validate='one_to_one');q.to_csv(p/(window+'_'+mode+'_daily.csv'),index=False)
                portfolios[name,window,mode]=q
                metrics.append(dict(method=name,window=window,signal=mode,days=len(ds),rows=len(subset),
                    date_equal_mse=dd.mse.mean(),date_equal_mae=dd.mae.mean(),common_mse=dd.common_mse.mean(),relative_mse=dd.relative_mse.mean(),
                    pooled_rmse=pooled['rmse'],pooled_mae=pooled['mae'],direction=m['direction']['accuracy'],
                    ic=m['cross_section']['ic']['mean'],rankic=m['cross_section']['rankic']['mean'],
                    **return_stats(port.top_net.to_numpy())))
            for k,kk in hd[(hd.method==name)&hd.date.isin(ds)].groupby('horizon'):
                by_horizon.append(dict(method=name,window=window,horizon=k,mse=kk.mse.mean(),mae=kk.mae.mean(),
                    pooled_rmse=np.sqrt(np.average(kk.mse,weights=kk.n)),direction=np.average(kk.direction,weights=kk.n)))
        print('EVALUATED '+name,flush=True)
    table=pd.DataFrame(metrics);table.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(by_horizon).to_csv(out/'by_horizon.csv',index=False)
    # Translation preserves IC/ranking and portfolio; Spearman invariant checked numerically.
    invariants={}
    for window in ['evaluation_86','full_226']:
        for mode in ['path','endpoint']:
            left=portfolios['A',window,mode];right=portfolios['Common',window,mode]
            for col in ['top_net','turnover','ic','rankic']:
                np.testing.assert_allclose(left[col],right[col],atol=1e-12,rtol=0)
            invariants[window+'_'+mode]=True
    draws=indices(86,5000,10,20261013);pairs=[];blocks=[]
    comparisons=[('Dual','A'),('Common','A'),('Individual','A'),('Dual','Common'),('Dual','Unsplit'),('Unsplit','A'),('PCA','A'),('DCT','A')]
    get=lambda name:daily[(daily.method==name)&daily.date.isin(evaluation)].sort_values('date').reset_index(drop=True)
    for left,right in comparisons:
        aa,bb=get(left),get(right)
        for metric in ['mse','mae','common_mse','relative_mse','path_direction','endpoint_direction']:
            diff=(aa[metric]-bb[metric]).to_numpy();boot=diff[draws].mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            pairs.append(dict(left=left,right=right,metric=metric,delta=diff.mean(),ci_low=lo,ci_high=hi,
                primary=(left,right,metric)==('Dual','A','mse')))
        for part,sl in [('first43',slice(0,43)),('last43',slice(43,None))]:
            a,b=aa.iloc[sl],bb.iloc[sl]
            blocks.append(dict(left=left,right=right,part=part,mse_delta=(a.mse-b.mse).mean(),
                relative_improvement=1-a.mse.mean()/b.mse.mean(),mae_delta=(a.mae-b.mae).mean()))
    paired=pd.DataFrame(pairs);paired.to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(blocks).to_csv(out/'time_block_metrics.csv',index=False)
    a=get('A');dual=get('Dual');common=get('Common');unsplit=get('Unsplit')
    primary=paired[paired.primary].iloc[0]
    halves=[(dual.mse.iloc[s]-a.mse.iloc[s]).mean() for s in [slice(0,43),slice(43,None)]]
    status='无收益'
    if dual.mse.mean()<a.mse.mean():
        status='MSE下降但MAE恶化' if dual.mae.mean()>a.mae.mean() else '初步有收益'
        if dual.mae.mean()<=a.mae.mean() and primary.ci_high<0 and max(halves)<0:status='较强的窗口内证据'
    verdict=dict(status=status,mse_relative_improvement=1-dual.mse.mean()/a.mse.mean(),
        mae_relative_improvement=1-dual.mae.mean()/a.mae.mean(),primary_ci=[primary.ci_low,primary.ci_high],
        half_mse_deltas=halves,dual_beats_common=bool(dual.mse.mean()<common.mse.mean()),
        dual_beats_unsplit=bool(dual.mse.mean()<unsplit.mse.mean()),
        selected_alphas={k:float(GRID[v]) for k,v in chosen.items()},
        independent_confirmation=False,primary_metric='equal-date trajectory MSE, final86dates')
    save(out/'verdict.json',verdict)
    make_report(out,table,daily,paired,verdict)
    save(out/'final_validation.json',dict(status='passed',common_ranking_portfolio_invariants=invariants,
        all_common_candidates_path_endpoint_rank_checks=rank_checks,nonzero_candidate_rank_invariance_passed=True,
        error_decomposition_exact=True,dual_addition_exact=True,methods=7,evaluation_days=86,
        future_label_poison='passed real guarded-label prefix replays; see future_poison_checks.json',
        primary_no_multiplicity_adjustment='one prespecified comparison; others descriptive'))
    print(table[table.window.eq('evaluation_86')].to_string(index=False),flush=True)
    print(json.dumps(verdict,ensure_ascii=False,indent=2),flush=True)


def make_report(out,table,daily,paired,verdict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for name in ['Common','Individual','Dual','Unsplit','PCA','DCT']:
        aa=daily[(daily.method==name)&daily.date.ge('2026-01-23')].sort_values('date')
        bb=daily[(daily.method=='A')&daily.date.ge('2026-01-23')].sort_values('date')
        axes[0].plot(pd.to_datetime(aa.date),np.cumsum(aa.mse.to_numpy()-bb.mse.to_numpy()),label=LABELS[name])
    axes[0].axhline(0,color='black',linewidth=.7);axes[0].set_title('累计 MSE 差：低于零优于 A',fontproperties=font)
    axes[0].legend(prop=font,fontsize=8);axes[0].tick_params(axis='x',rotation=30)
    selected=table[(table.window=='evaluation_86')&(table.signal=='path')].set_index('method').loc[list(LABELS)]
    base=selected.loc['A','date_equal_mse']
    yy=np.arange(len(selected));axes[1].barh(yy,(1-selected.date_equal_mse/base)*100)
    axes[1].set_yticks(yy,list(LABELS.values()),fontproperties=font);axes[1].axvline(0,color='black',linewidth=.7)
    axes[1].set_title('评估期轨迹 MSE 改善率（%）',fontproperties=font);axes[1].invert_yaxis()
    fig.tight_layout();fig.savefig(out/'prediction_loss_comparison.png',dpi=170);fig.savefig(out/'prediction_loss_comparison.svg');plt.close(fig)
    candidates=pd.read_csv(out/'all_candidate_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for ax,window,title in zip(axes,['selection_50','evaluation_86'],['选择期：所有非零候选均未超过零修正','评估期：事后诊断，不重新选参']):
        for name in ['Common','Individual','Dual_same_alpha','Unsplit']:
            q=candidates[(candidates.window==window)&(candidates.method==name)&(candidates.alpha>0)]
            ax.plot(q.alpha,100*q.mse_relative_improvement,marker='o',label=LABELS.get(name,'双头（相同步长）'))
        ax.set_xscale('log');ax.axhline(0,color='black',linewidth=.8)
        ax.set_title(title,fontproperties=font);ax.set_xlabel('更新率／步长系数',fontproperties=font)
        ax.set_ylabel('总 MSE 改善率（%）',fontproperties=font);ax.legend(prop=font)
    fig.tight_layout();fig.savefig(out/'nonzero_candidate_comparison.png',dpi=170);plt.close(fig)
    summary=[]
    for name in LABELS:
        p=table[(table.method==name)&(table.window=='evaluation_86')&(table.signal=='path')].iloc[0]
        ep=table[(table.method==name)&(table.window=='evaluation_86')&(table.signal=='endpoint')].iloc[0]
        summary.append({'方案':LABELS[name],'日期等权MSE':f'{p.date_equal_mse:.9f}',
            'MSE改善':f'{(1-p.date_equal_mse/base)*100:+.3f}%',
            'pooled RMSE':f'{p.pooled_rmse:.7f}','MAE':f'{p.pooled_mae:.7f}',
            '共同MSE':f'{p.common_mse:.9f}','相对MSE':f'{p.relative_mse:.9f}',
            'Path方向':f'{p.direction:.3%}','H10方向':f'{ep.direction:.3%}',
            'Path净收益':f'{p.net_cumulative_return:.3%}','Endpoint净收益':f'{ep.net_cumulative_return:.3%}'})
    rows=pd.DataFrame(summary)
    note=('主要评估为2026-01-23至2026-06-05的86个预测日；前50个选择日已按10日标签成熟清除重叠。'
        '共同头、个股头及未分离头分别选参；零步长合法，选到零即表示该分支未通过选择期。'
        '原HCA-PCA/DCT为历史联合更新结果，其他新头均不改变A；两种对比的含义不同。'
        '共同头不能改变截面排序或排序组合，因此其回测与A一致。'
        '回测为各报告窗口现金起始、Top30/Drop3/至少5日/单向15bps、次日收盘收益代理；86日收益不能直接与旧226日收益相比。'
        '全部模型预测相同股票键；标签为未来1至10日累计收益，误差不以价格元计。')
    lines=['# Prediction共同／个股残差分离实验','',f"评估结论：**{verdict['status']}**。",
        f"选择参数：{verdict['selected_alphas']}。",
        f"双头相对A的日期等权MSE改善率：{verdict['mse_relative_improvement']:.4%}；MAE改善率：{verdict['mae_relative_improvement']:.4%}。",
        f"主比较MSE差95%区间：{verdict['primary_ci']}（负数改善）。",'',note,'',rows.to_csv(index=False),
        '历史窗口已反复查看，结果是探索性证据；日期块区间不含重新训练与参数选择的不确定性。',
        '冻结基础表示、EMA、截面分解均为实验机制，不据此宣称算法原创。',
        '学习率尺度在前80日因果累计、之后固定；当前支持残差用更新前A重算，随后A自身更新可能使校准状态过时，本轮不对此增加额外机制。',
        '完整226日的事后选择方案仅作描述；部署可用的锁定参数从第140日开始评价。']
    lines.extend(['所有非零候选的选择期/评估期误差见all_candidate_metrics.csv；评估期数值仅作诊断，不回头修改选择。',
        '本次额外绘制nonzero_candidate_comparison.png，展示未通过选择的非零候选；零修正的退化置信区间[0,0]来自预测完全相同，不代表已估计某种改善。'])
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    style='body{max-width:1500px;margin:32px auto;padding:0 24px;font-family:Microsoft YaHei,Segoe UI,sans-serif;line-height:1.75;color:#213547}h1,h2{color:#087e8b}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#edf4f7}td:first-child{text-align:left}.note{padding:18px;background:#eff5f8;border-left:4px solid #087e8b}img{width:100%}'
    doc=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共同与个股残差分离实验</title><style>{style}</style><h1>共同／个股残差分离：预测效果验证</h1><p>2026-11-03 · Prediction专线 · 基础A不变 · PCA固定秩2</p>'
    doc+=f'<p class="note"><b>{html.escape(verdict["status"])}</b>：双头轨迹MSE改善{verdict["mse_relative_improvement"]:.3%}，MAE改善{verdict["mae_relative_improvement"]:.3%}。<br>选择参数：{html.escape(str(verdict["selected_alphas"]))}<br>主比较MSE差95%区间：{html.escape(str(verdict["primary_ci"]))}；负数代表改善。</p>'
    doc+='<h2>86日锁定评估</h2><div class="scroll">'+rows.to_html(index=False,border=0)+'</div><p>'+note+'</p><img src="prediction_loss_comparison.png" alt="累计预测损失差和MSE改善率">'
    doc+='<h2>如何归因</h2><p>总MSE＝共同MSE＋个股相对MSE。共同头负责所有股票同加一个10维修正；个股头截面均值为零。两个头各自训练，双头直接相加。未分离校准头用于辨别收益是否仅来自独立于A的校准。</p>'
    doc+='<h2>为什么选择零修正：全部预设步长</h2><p>以下非零步长结果用于诊断，不改变已锁定选择。Dual_same_alpha 指两个头使用同一步长的补充对照；正式双头分别按各自目标选参。选到零时，[0,0]置信区间仅反映预测完全相同。</p><img src="nonzero_candidate_comparison.png" alt="所有非零候选的MSE改善率"><div class="scroll">'+candidates.to_html(index=False,border=0,float_format=lambda x:f'{x:.9g}')+'</div>'
    doc+='<h2>配对区间</h2><div class="scroll">'+paired.to_html(index=False,border=0,float_format=lambda x:f'{x:.9g}')+'</div>'
    doc+='<h2>完整指标与历史描述</h2><div class="scroll">'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}')+'</div>'
    doc+='<p class="note">旧窗口已反复查看，属于探索性实验。块bootstrap：日期块10、5000次；不重训、不计参数选择不确定性。除预先指定的双头对A的MSE比较外，其余比较用于诊断。完整226日中早期使用事后选定参数，只作描述。没有独立确认或原创性认证。</p><p><a href="REPORT.md">实验说明</a> · <a href="metrics.csv">指标</a> · <a href="selection_scores.csv">选参</a> · <a href="training_checks.json">正确性</a> · <a href="verdict.json">结论</a></html>'
    (out/'comparison.html').write_text(doc,encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','report'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.mode=='run':run(args.out)
    report(args.out)
