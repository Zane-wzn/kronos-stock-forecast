"""Exploratory, chronologically purged CFUS screen; frozen real HCA features.

The historical test window is already inspected. This is NOT confirmation.
No optimizer other than coefficient-only ordinary SGD is used.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_delayed_feedback_pilot import Model, stable_stock_key

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'experiments/csi300_aligned'
OUT = DATA / 'models/cfus_round1_20261022'
SOURCES = {
    'Prediction': ('models/prediction_adapter/model.pt',
                   'models/prediction_adapter/predictions.csv',
                   '111f72975e77e165211b441a731b4da17dca4e26629b848839ce13aaffcd94eb'),
    'RankIC': ('../../.runtime/D_model_rankic_pilot_original.pt',
               'models/adapter_drankic_dynamic_static_20261013/predictions.csv',
               'fe1426d1ab0e627622860fd86eb81c8f05b74fbd54df29439556050a1d813961'),
}
ALPHAS = [0., .001, .003, .01, .03, .1, .3, 1.]
METHODS = ['Frozen', 'TargetPCA', 'ResidualPCA', 'DCT', 'Random', 'CFUS', 'Full']


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    w = pd.read_csv(DATA/'reference/date_windows.csv', dtype=str)
    w = w[w.split.eq('test')].sort_values('as_of_date').reset_index(drop=True)
    dates = w.as_of_date.tolist()
    assert len(dates) == 226
    # The fit and selection query labels MUST mature strictly before the next stage.
    fit = [i for i in range(11, 80) if w.iloc[i].label_end_date < dates[80]]
    selection = [i for i in range(80, 140) if w.iloc[i].label_end_date < dates[140]]
    evaluation = list(range(140, 226))
    cfg = dict(status='locked_before_efficacy', exploratory=True, previously_inspected_test=True,
        rank=2, horizon=10, alphas=ALPHAS, seed=20261022,
        fit_indices=fit, selection_indices=selection, evaluation_indices=evaluation,
        dates=dates, selection_stream_start=80, evaluation_start=140,
        boundary_rule='prior stage query label_end strictly before next stage query start',
        support='latest full path whose label_end is previous forecast day; 32 stable-hash stocks from historical preselected 60',
        support_loss='half squared error, mean over 32 stocks and 10 horizons',
        eta='alpha * H / mean fit-support augmented feature squared norm; per checkpoint; no label-based scaling',
        tuning='minimum equal-date MSE on selection; ties choose smaller alpha; same eight candidates including zero',
        one_step='reset coefficient head to zero each episode; update before next-day query',
        continuous='zero head at selection start; one update before each query; carry state through unscored embargo into evaluation',
        frozen='Kronos, Adapter, original head; no dropout, momentum, weight decay, clipping or normalization change',
        methods=METHODS, decision='screen: CFUS >=2% MSE reduction vs TargetPCA and tuned Full; MAE lower; first/last halves same direction; horizons 8-10 no >2% aggregate MSE worsening',
        uncertainty='paired circular date block bootstrap, length10, 4000 draws; descriptive, no multiplicity adjustment',
        limitations=['historical test reused; not independent confirmation',
                     'validation hidden absent; training hidden file truncated',
                     'full-path feedback only; cannot establish partial/delayed-mask-specific novelty',
                     'support stock rule simpler than historical F/P-intersection schedule; same across all present methods',
                     'CFUS learns a separate basis per positive step on fit only; same selection candidate count',
                     'fixed rank2 CFUS even if eigenvalue negative; no positive-eigenvalue gating'])
    dump(OUT/'locked_config.json', cfg)
    rows = []
    for name, ids in [('fit', fit), ('selection', selection), ('evaluation', evaluation)]:
        for q in ids:
            rows.append(dict(stage=name, query_index=q, query_date=dates[q],
                query_label_end=w.iloc[q].label_end_date, update_date=dates[q-1],
                support_origin=dates[q-11], support_label_end=w.iloc[q-11].label_end_date))
    pd.DataFrame(rows).to_csv(OUT/'split_episodes.csv', index=False)
    print(json.dumps({s:dict(days=sum(r['stage']==s for r in rows),
        start=next(r['query_date'] for r in rows if r['stage']==s),
        end=[r['query_date'] for r in rows if r['stage']==s][-1])
        for s in ['fit','selection','evaluation']}), flush=True)


def fixed_basis(a, r=2):
    _, v = np.linalg.eigh((a+a.T)/2)
    return v[:, -r:][:, ::-1].copy()


def losses(e):
    return np.mean(e*e, axis=0), np.mean(abs(e), axis=0)


def frame_rows(checkpoint, protocol, method, alpha, q, date, mse, mae):
    return [dict(checkpoint=checkpoint, protocol=protocol, method=method, alpha=alpha,
        query_index=q, date=date, horizon=k+1, mse=float(mse[k]), mae=float(mae[k]))
        for k in range(10)]


def make_data(name, cfg):
    ck, reference, expected = SOURCES[name]
    checkpoint = (DATA/ck).resolve()
    assert sha(checkpoint) == expected
    path = DATA/reference
    ref = pd.read_csv(path, dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
    assert len(ref) == 66103 and not ref.duplicated(['as_of_date','stock']).any()
    assert sorted(ref.as_of_date.unique()) == cfg['dates']
    model = Model().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    hh = np.load(DATA/'reference/hidden_cache/hidden_test.npy', mmap_mode='r')
    assert hh.shape == (66103,512) and np.isfinite(hh).all()
    z, pred = [], []
    with torch.no_grad():
        for start in range(0,len(hh),2048):
            zz = model.adapter(torch.tensor(np.array(hh[start:start+2048])))
            z.append(zz.numpy())
            pred.append(model.head(zz).numpy())
    z = np.column_stack([np.concatenate(z), np.ones(len(hh),dtype=np.float32)])
    pred = np.concatenate(pred).astype(np.float64)
    expected_pred = ref[[f'pred_return_{k}' for k in range(1,11)]].to_numpy()
    error = float(abs(pred-expected_pred).max())
    assert error < 1e-6, error
    y = ref[[f'actual_return_{k}' for k in range(1,11)]].to_numpy()
    assert np.isfinite(y).all()
    r = y-pred
    selected = json.loads((DATA/'models/hca_online_full_20261020/run_config.json').read_text())['training_stocks']
    groups = [g.index.to_numpy() for _,g in ref.groupby('as_of_date', sort=True)]
    supports = []
    for ids in groups:
        take = ref.loc[ids]
        take = take[take.stock.isin(selected)]
        take = take.assign(order=take.stock.map(stable_stock_key)).sort_values('order').iloc[:32]
        assert len(take) == 32
        supports.append(take.index.to_numpy())
    windows = pd.read_csv(DATA/'reference/date_windows.csv',dtype=str)
    windows = windows[windows.split.eq('test')].set_index('as_of_date')
    def episode(q):
        t, s = cfg['dates'][q-1], cfg['dates'][q-11]
        assert s < t < cfg['dates'][q]
        assert windows.loc[s,'label_end_date'] == t
        return supports[q-11], groups[q]
    # Precompute exact one-step gradients and responses, only legal support targets.
    ep = {}
    support_log = []
    for q in range(11,226):
        s, query = episode(q)
        zs = z[s].astype(np.float64)
        gradient = r[s].T @ zs / (len(s)*10)
        u = z[query].astype(np.float64) @ gradient.T
        ep[q] = (s, query, gradient, u)
        if q >= 80:
            for row in s:
                support_log.append(dict(query_date=cfg['dates'][q], update_date=cfg['dates'][q-1],
                    support_origin=cfg['dates'][q-11], support_label_end=cfg['dates'][q-1],
                    local_row=int(row), stock=ref.iloc[row].stock))
    logs = pd.DataFrame(support_log)
    assert not logs.local_row.duplicated().any()
    logs.to_csv(OUT/f'{name}_support_events.csv',index=False)
    norm = float(np.mean([np.mean(np.sum(z[ep[q][0]].astype(float)**2,axis=1)) for q in cfg['fit_indices']]))
    audit = dict(checkpoint=str(checkpoint), checkpoint_sha256=expected,
        reference_sha256=sha(path), frozen_prediction_max_abs_error=error,
        feature_mean_squared_norm=norm, rows=len(ref),
        feature_alignment='sort date,stock and verify all 66103x10 frozen outputs against archive',
        labels='archived actual_return_1..10 aligned by date,stock; original missing meta.csv not fabricated',
        continuous_updates=len(logs)//32, continuous_unique_support_rows=len(logs),
        continuous_supervised_scalars=len(logs)*10,
        future_support_reads=0, duplicate_supervision_keys=0)
    return z, r, ep, audit


def fit(z, r, ep, cfg, norm):
    cross = np.zeros((10,10)); energy=cross.copy()
    mean = np.zeros(10); residual_second=cross.copy()
    for q in cfg['fit_indices']:
        _, query, _, u = ep[q]
        e = r[query]
        cross += u.T @ e / len(e)
        energy += u.T @ u / len(e)
        mean += e.mean(axis=0)
        residual_second += e.T @ e / len(e)
    n = len(cfg['fit_indices'])
    cross /= n; energy /= n; mean /= n; residual_second /= n
    target = np.load(DATA/'models/hca_online_full_20261020/basis_pca.npy').astype(float)
    target = np.linalg.qr(target)[0]
    dct = np.cos(np.pi*(np.arange(10)[:,None]+.5)*np.arange(2)[None,:]/10)
    dct = np.linalg.qr(dct)[0]
    random = np.linalg.qr(np.random.default_rng(20261020).normal(size=(10,2)))[0]
    base = dict(TargetPCA=target, ResidualPCA=fixed_basis(residual_second-np.outer(mean,mean)),
                DCT=dct, Random=random, Full=np.eye(10))
    bases={}; spectra={}
    for a in ALPHAS:
        eta = a*10/norm
        m = eta*(cross+cross.T)/2 - eta**2*energy/2
        # alpha=0 ties all directions; use the smallest positive candidate's fit.
        if a == 0:
            eta0 = ALPHAS[1]*10/norm
            m = eta0*(cross+cross.T)/2 - eta0**2*energy/2
        b = fixed_basis(m)
        for method, basis in {**base,'CFUS':b,'Frozen':np.zeros((10,0))}.items():
            np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-12)
            bases[method,a] = basis
        spectra[str(a)] = dict(eigenvalues=np.linalg.eigvalsh(m)[::-1].tolist(),
            basis=b.tolist(), fit_half_mse_gain=float(np.trace(b.T@m@b)/10))
    return bases, spectra


def one_step(method, alpha, basis, eta, ids, r, ep, name, cfg):
    rows=[]
    for q in ids:
        _, query, _, u = ep[q]
        correction = eta*(u@basis)@basis.T
        e=r[query]-correction
        mse, mae = losses(e)
        rows.extend(frame_rows(name,'one_step',method,alpha,q,cfg['dates'][q],mse,mae))
    return rows


def continuous(method, alpha, basis, eta, start, stop, z, r, ep, name, cfg):
    # Equivalent coefficient-only SGD, expressed in output coordinates.
    # This output matrix always remains in col(B), and has exactly B K updates.
    k=np.zeros((basis.shape[1],z.shape[1]))
    rows=[]
    for q in range(start,stop):
        support, query, _, _ = ep[q]
        zs=z[support].astype(float)
        support_error=r[support]-(zs@k.T)@basis.T
        g=support_error.T@zs/(len(support)*10)
        k += eta*basis.T@g
        query_error=r[query]-(z[query].astype(float)@k.T)@basis.T
        if not np.isfinite(query_error).all():
            raise FloatingPointError((method,alpha,q))
        mse,mae=losses(query_error)
        rows.extend(frame_rows(name,'continuous',method,alpha,q,cfg['dates'][q],mse,mae))
    return rows


def correctness(z,r,ep,bases,norm,cfg):
    q=cfg['fit_indices'][0]
    support, query, g, u=ep[q]
    b=bases['CFUS',.03]; eta=.03*10/norm
    # Explicit autograd update on actual support samples, versus closed form.
    bt=torch.tensor(b,dtype=torch.float64)
    kt=torch.zeros((2,513),dtype=torch.float64,requires_grad=True)
    zs=torch.tensor(z[support],dtype=torch.float64)
    rr=torch.tensor(r[support],dtype=torch.float64)
    loss=.5*((zs@kt.T)@bt.T-rr).square().mean()
    loss.backward()
    got=(torch.tensor(z[query],dtype=torch.float64)@(-eta*kt.grad).T)@bt.T
    expected=eta*(u@b)@b.T
    autograd_error=float(abs(got.detach().numpy()-expected).max())
    assert autograd_error < 1e-12
    rot=np.linalg.qr(np.random.default_rng(100).normal(size=(2,2)))[0]
    rotation_error=float(abs(eta*(u@b@rot)@(b@rot).T-expected).max())
    assert rotation_error < 1e-12
    direct=.5*np.mean(np.sum(r[query]**2-(r[query]-expected)**2,axis=1))
    cross=u.T@r[query]/len(query); energy=u.T@u/len(query)
    m=eta*(cross+cross.T)/2-eta**2*energy/2
    trace_error=float(abs(direct-np.trace(b.T@m@b)))
    assert trace_error < 1e-12
    # Poison every label whose complete path is not yet available.
    poisoned=r.copy(); poisoned[query]=np.nan
    legal_gradient=poisoned[support].T@z[support].astype(float)/(len(support)*10)
    np.testing.assert_array_equal(legal_gradient,g)
    return dict(real_support_autograd_max_error=autograd_error,
        basis_rotation_max_error=rotation_error, utility_trace_error=trace_error,
        future_query_label_poison_leaves_support_update_identical=True,
        fit_last_label_before_selection=True, selection_last_label_before_evaluation=True)


def run():
    started=time.perf_counter()
    cfg=json.loads((OUT/'locked_config.json').read_text(encoding='utf-8'))
    assert cfg['alphas']==ALPHAS and cfg['methods']==METHODS
    assert not (OUT/'metrics.csv').exists()
    torch.set_num_threads(2)
    hh=DATA/'reference/hidden_cache/hidden_test.npy'
    assert sha(hh)=='20e91f77ac595f119885a99f7a3c7f96672403879e9a85213bd5b9d55f349310'
    selection_rows=[]; final_rows=[]; audits={}; chosen=[]; fit_rows=[]
    for name in SOURCES:
        print(json.dumps(dict(stage='preflight',checkpoint=name)),flush=True)
        z,r,ep,audit=make_data(name,cfg)
        norm=audit['feature_mean_squared_norm']
        bases,spectra=fit(z,r,ep,cfg,norm)
        dump(OUT/f'{name}_spectra.json',spectra)
        audit['correctness']=correctness(z,r,ep,bases,norm,cfg)
        for protocol in ['one_step','continuous']:
            print(json.dumps(dict(stage='selection',checkpoint=name,protocol=protocol)),flush=True)
            for method in METHODS:
                candidates=[]
                for alpha in ([0.] if method=='Frozen' else ALPHAS):
                    b=bases[method,alpha]; eta=alpha*10/norm
                    if protocol=='one_step':
                        result=one_step(method,alpha,b,eta,cfg['selection_indices'],r,ep,name,cfg)
                    else:
                        result=continuous(method,alpha,b,eta,80,max(cfg['selection_indices'])+1,z,r,ep,name,cfg)
                        result=[row for row in result if row['query_index'] in cfg['selection_indices']]
                    score=float(np.mean([row['mse'] for row in result]))
                    candidates.append((score,alpha))
                    selection_rows.extend(result)
                score,alpha=min(candidates)
                chosen.append(dict(checkpoint=name,protocol=protocol,method=method,
                    alpha=alpha,eta=alpha*10/norm,selection_mse=score,selection_candidates=len(candidates)))
                b=bases[method,alpha]; eta=alpha*10/norm
                np.save(OUT/f'{name}_{protocol}_{method}_basis.npy',b)
                if protocol=='one_step':
                    result=one_step(method,alpha,b,eta,cfg['evaluation_indices'],r,ep,name,cfg)
                    fit_rows.extend(one_step(method,alpha,b,eta,cfg['fit_indices'],r,ep,name,cfg))
                else:
                    result=continuous(method,alpha,b,eta,80,226,z,r,ep,name,cfg)
                    result=[row for row in result if row['query_index'] in cfg['evaluation_indices']]
                final_rows.extend(result)
                print(json.dumps(dict(checkpoint=name,protocol=protocol,method=method,alpha=alpha,selection_mse=score)),flush=True)
        audits[name]=audit
        del z,r,ep
    pd.DataFrame(selection_rows).to_csv(OUT/'selection_grid_daily_horizon.csv',index=False)
    pd.DataFrame(chosen).to_csv(OUT/'selected_steps.csv',index=False)
    pd.DataFrame(final_rows).to_csv(OUT/'evaluation_daily_horizon.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(OUT/'fit_selected_one_step_daily_horizon.csv',index=False)
    dump(OUT/'audit.json',dict(status='passed',hidden_sha256=sha(hh),checkpoints=audits,
         elapsed_seconds=time.perf_counter()-started))
    analyze()


def bootstrap(a,b,seed=20261022):
    rng=np.random.default_rng(seed); n=len(a)
    starts=rng.integers(n,size=(4000,(n+9)//10))
    idx=((starts[:,:,None]+np.arange(10))%n).reshape(4000,-1)[:,:n]
    values=100*(b[idx].mean(axis=1)-a[idx].mean(axis=1))/b[idx].mean(axis=1)
    return [float(v) for v in np.quantile(values,[.025,.975])]


def analyze():
    df=pd.read_csv(OUT/'evaluation_daily_horizon.csv')
    daily=df.groupby(['checkpoint','protocol','method','query_index','date'],as_index=False)[['mse','mae']].mean()
    daily.to_csv(OUT/'evaluation_daily.csv',index=False)
    metrics=daily.groupby(['checkpoint','protocol','method'],as_index=False)[['mse','mae']].mean()
    selected=pd.read_csv(OUT/'selected_steps.csv')
    metrics=metrics.merge(selected[['checkpoint','protocol','method','alpha','eta']],on=['checkpoint','protocol','method'])
    metrics.to_csv(OUT/'metrics.csv',index=False)
    blocks=[]; comparisons=[]; decisions=[]
    for (ck,proto),dd in daily.groupby(['checkpoint','protocol']):
        pivot=dd.pivot(index='query_index',columns='method',values='mse')
        pmae=dd.pivot(index='query_index',columns='method',values='mae')
        for competitor in METHODS:
            if competitor=='CFUS':continue
            a=pivot.CFUS.to_numpy(); b=pivot[competitor].to_numpy()
            ci=bootstrap(a,b)
            comparisons.append(dict(checkpoint=ck,protocol=proto,baseline=competitor,
                cfus_mse_reduction_pct=100*(b.mean()-a.mean())/b.mean(),
                ci95_low=ci[0],ci95_high=ci[1],
                cfus_mae_reduction_pct=100*(pmae[competitor].mean()-pmae.CFUS.mean())/pmae[competitor].mean(),
                cfus_daily_win_fraction=float((a<b).mean())))
        for label,ids in [('first43',pivot.index[:43]),('last43',pivot.index[43:])]:
            for method in METHODS:
                blocks.append(dict(checkpoint=ck,protocol=proto,block=label,method=method,
                    mse=float(pivot.loc[ids,method].mean()),mae=float(pmae.loc[ids,method].mean())))
        tests={}
        for base in ['TargetPCA','Full']:
            reduction=1-pivot.CFUS.mean()/pivot[base].mean()
            halves=[1-pivot.CFUS.iloc[s].mean()/pivot[base].iloc[s].mean() for s in [slice(0,43),slice(43,None)]]
            h=df[(df.checkpoint==ck)&(df.protocol==proto)&df.horizon.ge(8)].groupby('method').mse.mean()
            tests[base]=dict(mse_reduction_pct=float(100*reduction),
                mae_lower=bool(pmae.CFUS.mean()<pmae[base].mean()),
                halves_reduction_pct=[float(100*v) for v in halves],
                long_horizon_reduction_pct=float(100*(1-h.CFUS/h[base])),
                screen_pass=bool(reduction>=.02 and pmae.CFUS.mean()<pmae[base].mean()
                    and min(halves)>0 and h.CFUS<=1.02*h[base]))
        decisions.append(dict(checkpoint=ck,protocol=proto,comparisons=tests,
            screen_pass=all(v['screen_pass'] for v in tests.values())))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_block10_comparisons.csv',index=False)
    pd.DataFrame(blocks).to_csv(OUT/'half_metrics.csv',index=False)
    df.groupby(['checkpoint','protocol','method','horizon'],as_index=False)[['mse','mae']].mean().to_csv(OUT/'per_horizon.csv',index=False)
    dump(OUT/'decisions.json',dict(exploratory_only=True,decisions=decisions))
    print(metrics.to_string(index=False),flush=True)
    print(json.dumps(decisions,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run','analyze'])
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze}[args.stage]()
