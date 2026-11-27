#!/usr/bin/env python
from __future__ import annotations
import argparse,json,random,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler

def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
class DS(Dataset):
 def __init__(self,h,y):self.h=h;self.y=y
 def __len__(self):return len(self.h)
 def __getitem__(self,i):return torch.tensor(np.array(self.h[i]),dtype=torch.float32),torch.tensor(np.array(self.y[i]),dtype=torch.float32),i
class DateSampler(Sampler):
 def __init__(self,dates,b,sd,shuffle):
  self.g={};self.b=b;self.sd=sd;self.shuffle=shuffle
  for i,d in enumerate(dates):self.g.setdefault(d,[]).append(i)
 def __iter__(self):
  r=random.Random(self.sd);bs=[]
  for x in self.g.values():
   x=x.copy();r.shuffle(x) if self.shuffle else None;bs += [x[i:i+self.b] for i in range(0,len(x),self.b)]
  r.shuffle(bs) if self.shuffle else None;yield from bs
 def __len__(self):return sum((len(x)+self.b-1)//self.b for x in self.g.values())
class Adapter(nn.Module):
 def __init__(self,d,b,drop):super().__init__();self.ln=nn.LayerNorm(d);self.down=nn.Linear(d,b);self.drop=nn.Dropout(drop);self.up=nn.Linear(b,d);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,h):return h+self.up(self.drop(torch.nn.functional.gelu(self.down(self.ln(h)))))
class Model(nn.Module):
 def __init__(self,d,p,group,b,drop):super().__init__();self.adapter=Adapter(d,b,drop) if group in 'CD' else None;self.head=nn.Linear(d,p)
 def forward(self,h):return self.head(self.adapter(h) if self.adapter else h)
def rloss(p,y):
 p=p.mean(1);y=y.mean(1);dp=p[:,None]-p[None,:];dy=y[:,None]-y[None,:];m=dy!=0;return torch.nn.functional.softplus(-torch.sign(dy[m])*dp[m]).mean() if m.any() else p.sum()*0
def eval_model(model,h,y,dates,b,dev):
 model.eval();ls=[]
 with torch.no_grad():
  for hb,yb,_ in DataLoader(DS(h,y),batch_sampler=DateSampler(dates,b,1,False)):ls.append(float(torch.nn.functional.smooth_l1_loss(model(hb.to(dev)),yb.to(dev))))
 return float(np.mean(ls))
def metrics(df):
 r=[]
 for d,g in df.groupby('as_of_date'):
  q=max(1,len(g)//5);s=g.sort_values('pred_signal');r.append({'as_of_date':d,'n':len(g),'rankic':g.pred_signal.corr(g.actual_signal,method='spearman'),'ic':g.pred_signal.corr(g.actual_signal),'direction_accuracy':(g.pred_direction==g.actual_direction).mean(),'q5_q1':s.tail(q).actual_signal.mean()-s.head(q).actual_signal.mean()})
 return pd.DataFrame(r)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',choices=list('BCD'),required=True);p.add_argument('--cache-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--epochs',type=int,default=5);p.add_argument('--head-lr',type=float,default=3e-4);p.add_argument('--adapter-lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--lambda-rank',type=float,default=.2);p.add_argument('--seed',type=int,default=100);p.add_argument('--bottleneck',type=int,default=64);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--patience',type=int,default=2);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);started=time.time();seed(a.seed);c=Path(a.cache_dir)
 try:
  z={}
  for s in ['train','val','test']:
   z[s]=(np.load(c/f'hidden_{s}.npy',mmap_mode='r'),np.load(c/f'labels_{s}.npy',mmap_mode='r'),pd.read_csv(c/f'meta_{s}.csv'))
  d=z['train'][0].shape[1];model=Model(d,10,a.group,a.bottleneck,a.dropout).to(a.device);groups=[{'params':model.head.parameters(),'lr':a.head_lr}]
  if model.adapter:groups.append({'params':model.adapter.parameters(),'lr':a.adapter_lr})
  opt=torch.optim.AdamW(groups,weight_decay=a.weight_decay);best=None;bad=0;hist=[]
  th,ty,tm=z['train'];vh,vy,vm=z['val']
  for ep in range(1,a.epochs+1):
   model.train();ls=[];rs=[];dl=DataLoader(DS(th,ty),batch_sampler=DateSampler(tm.as_of_date.tolist(),a.batch_size,a.seed+ep,True))
   for hb,yb,ix in dl:
    assert len({tm.as_of_date.iloc[i] for i in ix.tolist()})==1;pr=model(hb.to(a.device));yy=yb.to(a.device);l=torch.nn.functional.smooth_l1_loss(pr,yy);r=rloss(pr,yy) if a.group=='D' else l*0;loss=l+a.lambda_rank*r;opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(float(l));rs.append(float(r))
   vl=eval_model(model,vh,vy,vm.as_of_date.tolist(),a.batch_size,a.device);row={'epoch':ep,'train_huber':np.mean(ls),'train_rank':np.mean(rs),'val_huber':vl};hist.append(row);print(f'[TRAIN {a.group}] {row}',flush=True)
   if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep);bad=0
   else:bad+=1
   if bad>=a.patience:break
  model.load_state_dict(best[1]);eh,ey,em=z['test'];model.eval();rows=[]
  with torch.no_grad():
   for i in range(0,len(eh),a.batch_size):
    pr=model(torch.tensor(np.array(eh[i:i+a.batch_size]),device=a.device)).cpu().numpy();yy=np.array(ey[i:i+a.batch_size]);mm=em.iloc[i:i+len(pr)]
    for (_,m),r,y in zip(mm.iterrows(),pr,yy):
     ps=float(r.mean());ac=float(y.mean());rows.append({'group':a.group,'seed':a.seed,'as_of_date':m.as_of_date,'stock':m.stock,'pred_signal':ps,'actual_signal':ac,'pred_direction':int(ps>0),'actual_direction':int(ac>0),**{f'pred_return_{j+1}':r[j] for j in range(10)},**{f'actual_return_{j+1}':y[j] for j in range(10)}})
  df=pd.DataFrame(rows);df.to_csv(out/'predictions.csv',index=False);md=metrics(df);md.to_csv(out/'metrics_by_date.csv',index=False);pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False);torch.save(best[1],out/'model.pt')
  sm={'status':'complete','group':a.group,'seed':a.seed,'batch_size':a.batch_size,'best_epoch':best[2],'train_rows':len(th),'val_rows':len(vh),'test_rows':len(eh),'test_dates':df.as_of_date.nunique(),'rankic_mean':md.rankic.mean(),'ic_mean':md.ic.mean(),'direction_accuracy':(df.pred_direction==df.actual_direction).mean(),'q5_q1_mean':md.q5_q1.mean(),'elapsed_seconds':time.time()-started};(out/'summary.json').write_text(json.dumps(sm,indent=2),encoding='utf-8');print(json.dumps(sm,indent=2),flush=True)
 except Exception as e:(out/'FAILED.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2),encoding='utf-8');raise
if __name__=='__main__':main()
