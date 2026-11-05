"""Locked exploratory spectral screen. No historical artifacts are overwritten.
Frozen Kronos/Adapter/base head, coefficient-only SGD: isolates output geometry.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, hashlib, json, time, html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/csi300_aligned'
OLD=DATA/'models/cfus_round1_20261022'
ATTR=DATA/'models/hca_gain_attribution_20261028'
OUT=DATA/'models/spectral_validation_20261028'
ALPHAS=[0.,.001,.003,.01,.03,.1,.3,1.]
METHODS=['Frozen','TargetPCA2','DCT2','DCT4','HighDCT2','Random2','SpectralShrink','Full']
SOURCES={'Prediction':('models/prediction_adapter/model.pt','models/prediction_adapter/predictions.csv'),
'RankIC':('../../.runtime/D_model_rankic_pilot_original.pt','models/adapter_drankic_dynamic_static_20261013/predictions.csv')}
PC=[f'pred_return_{k}' for k in range(1,11)]
AC=[f'actual_return_{k}' for k in range(1,11)]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def key(s):return hashlib.sha256(('prefix-pilot-v1|'+str(s)).encode()).hexdigest()
class Adapter(nn.Module):
 def __init__(self):
  super().__init__();self.ln=nn.LayerNorm(512);self.down=nn.Linear(512,64);self.up=nn.Linear(64,512)
 def forward(self,h):return h+self.up(F.gelu(self.down(self.ln(h))))
class Model(nn.Module):
 def __init__(self):super().__init__();self.adapter=Adapter();self.head=nn.Linear(512,10)
def dct():
 d=np.cos(np.pi*(np.arange(10)[:,None]+.5)*np.arange(10)[None,:]/10)
 d/=np.linalg.norm(d,axis=0)
 np.testing.assert_allclose(d.T@d,np.eye(10),atol=1e-14)
 return d
D=dct()
def prepare():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 cfg=json.loads((OLD/'locked_config.json').read_text(encoding='utf-8'))
 cfg.update(status='locked_before_new_efficacy',experiment='spectral_validation',methods=METHODS,
  seed=20261028,alphas=ALPHAS,
  hypothesis='low-frequency error concentration does not imply predictive update utility; test constrained and soft-shrunk updates',
  geometry={'TargetPCA2':'saved training rank2','DCT2':'frequencies 0,1','DCT4':'frequencies 0..3',
   'HighDCT2':'frequencies 8,9 negative control','Random2':'fixed seed20261020 rank2',
   'SpectralShrink':'10 frequency preconditioner weights 1/(1+(k/2)^2), normalized trace10; fixed cutoff, no extra tuning',
   'Full':'identity output update'},
  protocols=['continuous'],
  diagnostic='date-equal squared-error frequency energy including mean bias; separate from centered target covariance',
  budget='all methods 146 unique updates/46720 scalar labels, same 32 stocks/event; scoring 86 dates; no refit on evaluation',
  selection='same 8 learning rates including zero; minimum selection MSE, ties smaller; states start at q80 and carry through unscored embargo',
  screen='mechanism screen only: candidate must improve MSE and MAE vs tuned Full and PCA in both checkpoints, each half same direction; 95% CI vs Full excludes zero for strong evidence',
  limitations=['already inspected CSI300 window; exploratory only','frozen representation/base head and ordinary SGD, not full HCA AdamW',
   'rank4/full/shrink parameter capacity differs; learning rate tuned equally; no portfolio return claim',
   'one random seed; no multiplicity adjustment; block bootstrap resamples realized prediction losses, not retraining',
   'full mature feedback only; fixed panel may have survivor/constituent bias'])
 dump(OUT/'locked_config.json',cfg)
 dump(OUT/'source_manifest.json',{'script_sha256':sha(Path(__file__)), 'old_config_sha256':sha(OLD/'locked_config.json'),
  'old_audit_sha256':sha(OLD/'audit.json'),'old_support_sha256':{n:sha(OLD/f'{n}_support_events.csv') for n in SOURCES}})
 print(json.dumps({'prepared':str(OUT),'methods':METHODS,'evaluation_days':len(cfg['evaluation_indices'])}),flush=True)
def load(name,cfg):
 prior=json.loads((OLD/'audit.json').read_text())
 ck,ref=SOURCES[name];ck=(DATA/ck).resolve();ref=DATA/ref
 assert sha(ck)==prior['checkpoints'][name]['checkpoint_sha256']
 assert sha(ref)==prior['checkpoints'][name]['reference_sha256']
 frame=pd.read_csv(ref,dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
 assert len(frame)==66103 and not frame.duplicated(['as_of_date','stock']).any()
 assert sorted(frame.as_of_date.unique())==cfg['dates']
 model=Model().eval();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
 hidden=np.load(DATA/'reference/hidden_cache/hidden_test.npy',mmap_mode='r')
 zz=[];pp=[]
 with torch.no_grad():
  for i in range(0,len(hidden),2048):
   z=model.adapter(torch.tensor(np.array(hidden[i:i+2048])));zz.append(z.numpy());pp.append(model.head(z).numpy())
 z=np.column_stack([np.concatenate(zz),np.ones(len(frame),dtype=np.float32)]).astype(float)
 pred=np.concatenate(pp).astype(float);err=float(abs(pred-frame[PC].to_numpy()).max());assert err<1e-6
 y=frame[AC].to_numpy();r=y-pred
 groups=[g.index.to_numpy() for _,g in frame.groupby('as_of_date',sort=True)]
 events=pd.read_csv(OLD/f'{name}_support_events.csv',dtype={'stock':str})
 assert not events.local_row.duplicated().any() and len(events)==4672
 supports={}
 for date,ev in events.groupby('query_date',sort=True):
  q=cfg['dates'].index(date);ids=ev.local_row.to_numpy()
  assert len(ids)==32 and (ev.support_label_end<date).all()
  assert (ev.support_label_end==cfg['dates'][q-1]).all()
  assert (ev.support_origin==cfg['dates'][q-11]).all()
  assert np.array_equal(frame.iloc[ids].stock.to_numpy(),ev.stock.to_numpy())
  assert (frame.iloc[ids].as_of_date==cfg['dates'][q-11]).all()
  supports[q]=ids
 # Fit scaling uses only historical training-stock availability and already matured fit supports.
 meta=pd.read_csv(DATA/'doubleadapt_cache_v2/meta.csv',dtype={'stock':str})
 stocks=sorted(meta.loc[meta.split.eq('train'),'stock'].unique(),key=key)[:60]
 fit_support=[]
 for q in cfg['fit_indices']:
  ids=[i for i in groups[q-11] if frame.iloc[i].stock in stocks];ids=sorted(ids,key=lambda i:key(frame.iloc[i].stock))[:32]
  assert len(ids)==32;fit_support.append(ids)
 norm=float(np.mean([np.mean(np.sum(z[s]**2,axis=1)) for s in fit_support]))
 assert abs(norm-prior['checkpoints'][name]['feature_mean_squared_norm'])<1e-10
 return frame,z,r,y,groups,supports,norm,{'prediction_reproduction_max_error':err,'norm':norm,'checkpoint_sha256':sha(ck),'reference_sha256':sha(ref)}
def geometries():
 p=np.linalg.qr(np.load(DATA/'models/hca_online_full_20261020/basis_pca.npy').astype(float))[0]
 random=np.linalg.qr(np.random.default_rng(20261020).normal(size=(10,2)))[0]
 weights=1/(1+(np.arange(10)/2)**2);weights*=10/weights.sum()
 b={'TargetPCA2':p,'DCT2':D[:,:2],'DCT4':D[:,:4],'HighDCT2':D[:,8:],'Random2':random}
 out={m:x@x.T for m,x in b.items()};out.update(Frozen=np.zeros((10,10)),Full=np.eye(10),SpectralShrink=(D*weights)@D.T)
 np.save(OUT/'dct_basis.npy',D);np.save(OUT/'spectral_shrink_weights.npy',weights)
 return out
def replay(z,r,groups,supports,p,eta,start=80,stop=226):
 w=np.zeros((10,513));rows=[];harm=[]
 for q in range(start,stop):
  s=supports[q];query=groups[q];zs=z[s];zq=z[query]
  before=r[query]-zq@w.T # diagnostic only, never used by update
  gradient=(r[s]-zs@w.T).T@zs/(len(s)*10)
  w+=eta*p@gradient
  after=r[query]-zq@w.T
  if not np.isfinite(after).all():raise FloatingPointError(q)
  rows.extend([dict(query_index=q,horizon=k+1,mse=float(np.mean(after[:,k]**2)),mae=float(np.mean(abs(after[:,k])))) for k in range(10)])
  harm.append(dict(query_index=q,pre_update_query_mse=float(np.mean(before**2)),post_update_query_mse=float(np.mean(after**2))))
 return rows,harm,w

def correctness(z,r,g,s,p,norm):
 q=80;ids=s[q];eta=.03*10/norm;b=D[:,:2]
 kt=torch.zeros((2,513),dtype=torch.float64,requires_grad=True)
 zs=torch.tensor(z[ids]);bt=torch.tensor(b);rr=torch.tensor(r[ids]);loss=.5*((zs@kt.T)@bt.T-rr).square().mean();loss.backward()
 expected=(-eta*kt.grad.detach().numpy());got=eta*b.T@(r[ids].T@z[ids]/320)
 error=float(abs(expected-got).max());assert error<1e-12
 full_error=float(abs(D@D.T-np.eye(10)).max());assert full_error<1e-14
 a,_,wa=replay(z,r,g,s,p['SpectralShrink'],eta,stop=100)
 poison=r.copy();allowed=np.concatenate([s[x] for x in range(80,100)]);mask=np.ones(len(r),bool);mask[allowed]=False;poison[mask]=np.nan
 # Isolate the training path: no query-label accesses allowed here.
 def train_only(target):
  w=np.zeros((10,513))
  for x in range(80,100):
   ids=s[x];w+=eta*p['SpectralShrink']@((target[ids]-z[ids]@w.T).T@z[ids]/320)
  return w
 wp=train_only(poison);assert np.array_equal(wa,wp)
 _,_,wb=replay(z,r,g,s,D@D.T,eta,stop=100)
 _,_,wc=replay(z,r,g,s,p['Full'],eta,stop=100)
 equivalent=float(abs(wb-wc).max());assert equivalent<1e-12
 return {'autograd_DCT2_max_error':error,'DCT_orthogonality_error':full_error,'full_DCT_vs_identity_parameter_error':equivalent,'future_label_poison_training_identical':True}
def boot(a,b,seed=20261028):
 rng=np.random.default_rng(seed);n=len(a);starts=rng.integers(n,size=(4000,(n+9)//10));idx=((starts[:,:,None]+np.arange(10))%n).reshape(4000,-1)[:,:n]
 v=100*(b[idx].mean(1)-a[idx].mean(1))/b[idx].mean(1)
 return np.quantile(v,[.025,.975]).tolist()
def diagnostics(name,frame,r,y,groups,cfg):
 rows=[];pca=np.linalg.qr(np.load(DATA/'models/hca_online_full_20261020/basis_pca.npy'))[0]
 arrays={'Frozen':r}
 for method in ['A','HCA']:
  f=pd.read_csv(ATTR/f'{name}_Live_{method}'/'causal_predictions.csv',dtype={'stock':str}).sort_values(['as_of_date','stock']).reset_index(drop=True)
  assert np.array_equal(f[['as_of_date','stock']].to_numpy(),frame[['as_of_date','stock']].to_numpy())
  arrays[method]=y-f[PC].to_numpy()
 for method,e in arrays.items():
  for q,ids in enumerate(groups):
   power=np.mean((e[ids]@D)**2,axis=0)
   for k in range(10):rows.append(dict(checkpoint=name,method=method,query_index=q,date=cfg['dates'][q],frequency=k,error_energy=float(power[k]),total_energy=float(power.sum())))
 pd.DataFrame(rows).to_csv(OUT/f'{name}_frequency_daily.csv',index=False)
 # Causal 40-date label-covariance diagnostic, available before each forecast q.
 drift=[]
 for q in range(50,226,5):
  origin_ids=list(range(max(0,q-50),q-10));mu=np.mean([y[groups[o]].mean(0) for o in origin_ids],axis=0)
  second=np.mean([y[groups[o]].T@y[groups[o]]/len(groups[o]) for o in origin_ids],axis=0);cov=second-np.outer(mu,mu)
  val,v=np.linalg.eigh(cov);new=v[:,-2:];sv=np.linalg.svd(pca.T@new,compute_uv=False);angles=np.degrees(np.arccos(np.clip(sv,-1,1)))
  drift.append(dict(checkpoint=name,query_index=q,date=cfg['dates'][q],latest_origin=cfg['dates'][q-11],
   PCA2_covariance_coverage=float(np.trace(pca.T@cov@pca)/np.trace(cov)),DCT2_covariance_coverage=float(np.trace(D[:,:2].T@cov@D[:,:2])/np.trace(cov)),
   optimal2_coverage=float(val[-2:].sum()/val.sum()),max_subspace_angle_deg=float(max(angles))))
 pd.DataFrame(drift).to_csv(OUT/f'{name}_basis_drift.csv',index=False)
def run_all():
 started=time.perf_counter();cfg=json.loads((OUT/'locked_config.json').read_text());assert not (OUT/'metrics.csv').exists()
 assert cfg['methods']==METHODS;torch.set_num_threads(2)
 hh=DATA/'reference/hidden_cache/hidden_test.npy';assert sha(hh)=='20e91f77ac595f119885a99f7a3c7f96672403879e9a85213bd5b9d55f349310'
 p=geometries();selection=[];chosen=[];evals=[];harms=[];audits={}
 for name in SOURCES:
  print('loading '+name,flush=True);f,z,r,y,g,s,norm,audit=load(name,cfg)
  audit['correctness']=correctness(z,r,g,s,p,norm);diagnostics(name,f,r,y,g,cfg)
  for m in METHODS:
   choices=[]
   for alpha in ([0.] if m=='Frozen' else ALPHAS):
    rr,_,_=replay(z,r,g,s,p[m],alpha*10/norm,stop=max(cfg['selection_indices'])+1)
    rr=[x for x in rr if x['query_index'] in cfg['selection_indices']]
    score=float(np.mean([x['mse'] for x in rr]));choices.append((score,alpha));selection.append(dict(checkpoint=name,method=m,alpha=alpha,selection_mse=score))
   score,alpha=min(choices);chosen.append(dict(checkpoint=name,method=m,alpha=alpha,eta=alpha*10/norm,selection_mse=score))
   rr,hh,ww=replay(z,r,g,s,p[m],alpha*10/norm)
   for x in rr:
    if x['query_index'] in cfg['evaluation_indices']:evals.append(dict(checkpoint=name,method=m,date=cfg['dates'][x['query_index']],**x))
   for x in hh:
    if x['query_index'] in cfg['evaluation_indices']:harms.append(dict(checkpoint=name,method=m,**x))
   np.save(OUT/f'{name}_{m}_final_correction.npy',ww)
   print(json.dumps(dict(checkpoint=name,method=m,alpha=alpha,selection_mse=score)),flush=True)
  audits[name]=audit
 pd.DataFrame(selection).to_csv(OUT/'selection_grid.csv',index=False);pd.DataFrame(chosen).to_csv(OUT/'selected_steps.csv',index=False)
 pd.DataFrame(evals).to_csv(OUT/'evaluation_daily_horizon.csv',index=False);pd.DataFrame(harms).to_csv(OUT/'update_utility_daily.csv',index=False)
 dump(OUT/'audit.json',dict(status='passed',checkpoints=audits,elapsed_seconds=time.perf_counter()-started,updates_per_method=146,scalar_labels_per_method=46720,duplicate_supports=0))
 analyze()
def analyze():
 cfg=json.loads((OUT/'locked_config.json').read_text());df=pd.read_csv(OUT/'evaluation_daily_horizon.csv')
 daily=df.groupby(['checkpoint','method','query_index','date'],as_index=False)[['mse','mae']].mean();daily.to_csv(OUT/'evaluation_daily.csv',index=False)
 metrics=daily.groupby(['checkpoint','method'],as_index=False)[['mse','mae']].mean().merge(pd.read_csv(OUT/'selected_steps.csv'),on=['checkpoint','method'])
 prior=pd.read_csv(OLD/'metrics.csv');reproduction=[]
 for name in SOURCES:
  for current,old in [('Frozen','Frozen'),('TargetPCA2','TargetPCA'),('DCT2','DCT'),('Full','Full'),('Random2','Random')]:
   now=metrics[(metrics.checkpoint==name)&(metrics.method==current)].iloc[0];prev=prior[(prior.checkpoint==name)&(prior.protocol=='continuous')&(prior.method==old)].iloc[0]
   error=abs(now.mse-prev.mse);assert error<1e-12,(name,current,error)
   assert now.alpha==prev.alpha
   reproduction.append(dict(checkpoint=name,method=current,mse_error=float(error)))
 dump(OUT/'historical_reproduction.json',dict(status='passed',rows=reproduction))
 comparisons=[]
 for name in SOURCES:
  dd=daily[daily.checkpoint==name];pv=dd.pivot(index='query_index',columns='method',values='mse');ma=dd.pivot(index='query_index',columns='method',values='mae')
  for m in METHODS:
   for base in ['Frozen','TargetPCA2','Full']:
    if m==base:continue
    a=pv[m].to_numpy();b=pv[base].to_numpy();ci=boot(a,b)
    h=df[(df.checkpoint==name)&df.horizon.ge(8)].groupby('method').mse.mean()
    comparisons.append(dict(checkpoint=name,method=m,baseline=base,mse_reduction_pct=float(100*(1-a.mean()/b.mean())),ci95_low=ci[0],ci95_high=ci[1],mae_reduction_pct=float(100*(1-ma[m].mean()/ma[base].mean())),first43_reduction_pct=float(100*(1-a[:43].mean()/b[:43].mean())),last43_reduction_pct=float(100*(1-a[43:].mean()/b[43:].mean())),long_horizon_reduction_pct=float(100*(1-h[m]/h[base]))))
 metrics.to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
 bands=[]
 for name in SOURCES:
  freq=pd.read_csv(OUT/f'{name}_frequency_daily.csv')
  for method,dd in freq.groupby('method'):
   for label,ids in [('fit',cfg['fit_indices']),('selection',cfg['selection_indices']),('eval_first43',cfg['evaluation_indices'][:43]),('eval_last43',cfg['evaluation_indices'][43:]),('evaluation',cfg['evaluation_indices'])]:
    power=dd[dd.query_index.isin(ids)].groupby('frequency').error_energy.mean();total=power.sum()
    bands.append(dict(checkpoint=name,method=method,stage=label,low2_pct=float(100*power.iloc[:2].sum()/total),low4_pct=float(100*power.iloc[:4].sum()/total),high6_pct=float(100*power.iloc[4:].sum()/total)))
 pd.DataFrame(bands).to_csv(OUT/'frequency_summary.csv',index=False)
 utility=pd.read_csv(OUT/'update_utility_daily.csv');utility['harmful']=utility.post_update_query_mse>utility.pre_update_query_mse;utility['relative_gain_pct']=100*(1-utility.post_update_query_mse/utility.pre_update_query_mse)
 utility.groupby(['checkpoint','method'],as_index=False)[['harmful','relative_gain_pct']].mean().to_csv(OUT/'update_utility_summary.csv',index=False)
 print(metrics.to_string(index=False),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','analyze']);args=parser.parse_args();{'prepare':prepare,'run':run_all,'analyze':analyze}[args.stage]()
