
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def stamps(dates):
 d=pd.Series(pd.to_datetime(dates)); return pd.DataFrame({'minute':d.dt.minute,'hour':d.dt.hour,'weekday':d.dt.weekday,'day':d.dt.day,'month':d.dt.month}).to_numpy(np.float32)
def load_predictor(repo,model_name,tokenizer_name,device):
 import sys
 sys.path.insert(0,str(Path(repo).resolve()))
 from model import Kronos,KronosPredictor,KronosTokenizer
 tok=KronosTokenizer.from_pretrained(tokenizer_name); model=Kronos.from_pretrained(model_name); pred=KronosPredictor(model,tok,device=device,max_context=512); model.eval(); tok.eval()
 return pred
def main():
 p=argparse.ArgumentParser(); p.add_argument('--panel-dir',required=True); p.add_argument('--cache-dir',required=True); p.add_argument('--kronos-repo',required=True); p.add_argument('--model-name',default='NeoQuasar/Kronos-small'); p.add_argument('--tokenizer-name',default='NeoQuasar/Kronos-Tokenizer-base'); p.add_argument('--device',default='cuda:0'); p.add_argument('--split',default='test',choices=['train','val','test']); p.add_argument('--max-stocks',type=int); p.add_argument('--max-dates',type=int); p.add_argument('--smoke',action='store_true'); p.add_argument('--output',required=True); a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 pair=pd.read_csv(Path(a.panel_dir)/'eligible_stock_date_pairs.csv'); windows=pd.read_csv(Path(a.panel_dir)/'date_windows.csv'); meta=pair[pair.split.eq(a.split)].copy()
 if a.max_stocks: meta=meta[meta.stock.isin(sorted(meta.stock.unique())[:a.max_stocks])].copy()
 dates=sorted(meta.as_of_date.unique());
 if a.max_dates: dates=dates[:a.max_dates] if a.split=='test' else dates[-a.max_dates:]; meta=meta[meta.as_of_date.isin(dates)].copy()
 if a.smoke: dates=dates[:2]; meta=meta[meta.as_of_date.isin(dates)].copy()
 meta=meta.sort_values(['as_of_date','stock']).reset_index(drop=True); win=windows[windows.split.eq(a.split)].drop_duplicates('as_of_date').set_index('as_of_date'); pred=load_predictor(a.kronos_repo,a.model_name,a.tokenizer_name,a.device); d=int(pred.model.d_model); n=len(meta); seq=np.lib.format.open_memmap(out/f'hidden_seq_{a.split}.npy',mode='w+',dtype='float32',shape=(n,90,d)); last=np.lib.format.open_memmap(out/f'hidden_{a.split}.npy',mode='w+',dtype='float32',shape=(n,d)); row=0
 with torch.no_grad():
  for date,g in meta.groupby('as_of_date',sort=True):
   w=win.loc[date]; inp=pd.to_datetime(str(w.input_dates).split('|')); stocks=g.stock.astype(str).tolist()
   for start in range(0,len(stocks),16):
    xs=[]; ts=[]
    for stock in stocks[start:start+16]:
     frame=pd.read_csv(Path(ROOT)/'data/csi300_daily/csi300_daily'/f'{stock.replace(".","_")}.csv'); frame['date']=pd.to_datetime(frame['date']).dt.normalize(); frame=frame.set_index('date'); c=frame.loc[inp]; cols=['open','high','low','close','volume','amount'];
     if 'amount' not in c: c=c.copy(); c['amount']=c[['open','high','low','close']].mean(axis=1)*c['volume']; x=c[cols].to_numpy(np.float32); x=(x-x.mean(0,keepdims=True))/(x.std(0,keepdims=True)+1e-5); xs.append(np.clip(x,-5,5)); ts.append(stamps(inp))
    toks=pred.tokenizer.encode(torch.as_tensor(np.stack(xs),device=pred.device),half=True); _,ctx=pred.model.decode_s1(toks[0],toks[1],torch.as_tensor(np.stack(ts),device=pred.device)); arr=ctx.float().cpu().numpy(); m=len(arr); seq[row:row+m]=arr; last[row:row+m]=arr[:,-1]; row+=m
 seq.flush(); last.flush(); meta.to_csv(out/f'meta_{a.split}.csv',index=False); json.dump({'status':'complete','split':a.split,'shape':list(seq.shape),'dtype':'float32','last_token_max_abs_error':float(np.max(np.abs(np.asarray(seq)[:,-1]-np.asarray(last)))),'model':a.model_name,'tokenizer':a.tokenizer_name,'hidden_sha256':sha256(out/f'hidden_{a.split}.npy'),'sequence_sha256':sha256(out/f'hidden_seq_{a.split}.npy')},open(out/f'sequence_cache_{a.split}.json','w'),indent=2)
if __name__=='__main__': main()
