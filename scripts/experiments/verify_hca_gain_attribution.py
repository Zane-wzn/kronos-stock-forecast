from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from scripts.experiments.run_hca_gain_attribution import old,full
out=ROOT/'experiments/csi300_aligned/models/hca_gain_attribution_20261028'
status=json.loads((out/'status.json').read_text()); assert status['status']=='complete' and status['streams']==20
au=json.loads((out/'stream_audits.json').read_text()); assert len(au)==20 and all(x['status']=='passed' for x in au)
assert sum(x['updates'] for x in au)==4320 and sum(x['labels'] for x in au)==1382400
assert [x['historical_max_abs_error'] for x in au if x['historical_max_abs_error'] is not None]==[0.]*4
m=pd.read_csv(out/'attribution_metrics.csv'); b=pd.read_csv(out/'attribution_bootstrap.csv')
assert len(m)==40 and not m.duplicated(['name','signal']).any(); assert len(b)==1064
assert np.isfinite(b[['delta','ci_low','ci_high']]).all().all()
assert (b.ci_low<=b.ci_high).all()
for name in m.name.unique():
 u=pd.read_csv(out/name/'update_log.csv'); a=pd.read_csv(out/name/'access_log.csv')
 assert len(u)==216 and len(a)==69120 and not a.duplicated(['row_id','horizon']).any()
 if 'MatchedStep' in name: assert u.matched_norm_abs_error.max()<2e-8
 if '_Frozen_' in name: assert u.adapter_update_norm.eq(0).all()
 p=pd.read_csv(out/name/'predictions.csv',dtype={'stock':str})
 assert len(p)==66103 and p.method.eq(name).all()
 # Exact same metadata events across methods, ignoring prediction hash which must differ.
 events=a[['update_date','row_id','horizon','origin_date','label_date','complete_date','stock']]
 if name==m.name.iloc[0]: canonical=events
 else: pd.testing.assert_frame_equal(events,canonical)
class Tables(HTMLParser):
 def __init__(self): super().__init__(); self.tables=[]; self.current=None; self.row=None
 def handle_starttag(self,tag,attrs):
  if tag=='table': self.current=[]
  if tag=='tr': self.row=0
  if tag in ('th','td'): self.row+=1
 def handle_endtag(self,tag):
  if tag=='tr': self.current.append(self.row)
  if tag=='table': self.tables.append(self.current); self.current=None
parser=Tables(); parser.feed((out/'hca_gain_attribution.html').read_text(encoding='utf8'))
assert all(len(set(t))==1 for t in parser.tables); assert len(parser.tables[0])==9
inputs=json.loads((out/'input_manifest.json').read_text())
for ck,e in full.MODELS.items(): assert old.sha256(e['checkpoint'])==inputs[ck]['checkpoint_sha256']
report={'status':'passed','streams':20,'metric_rows':40,'bootstrap_rows':len(b),'updates':4320,'label_accesses':1382400,'historical_prediction_max_error':0.,'html_table_dimensions':[[len(t),t[0]] for t in parser.tables],'matched_max_norm_error':float(pd.read_csv(out/'attribution_training_summary.csv').matched_norm_abs_error_max.max()),'all_stream_event_schedules_identical':True,'input_checkpoints_unchanged':True}
old.atomic_json(out/'final_validation.json',report)
paths=[p for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json']
paths += [out/name/file for name in m.name.unique() for file in ('causal_predictions.csv','predictions.csv','update_log.csv','access_log.csv')]
old.atomic_json(out/'artifact_manifest.json',{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':old.sha256(p)} for p in paths})
print(json.dumps(report))
