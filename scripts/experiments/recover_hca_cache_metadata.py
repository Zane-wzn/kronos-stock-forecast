import json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from src.doubleadapt.data import sha256,validate_test
root=Path.cwd(); cache=root/'experiments/csi300_aligned/doubleadapt_cache_v2'; pair=root/'experiments/csi300_aligned/reference/eligible_stock_date_pairs.csv'
expected=json.loads((cache/'manifest.json').read_text())['pairs_sha256']
assert sha256(pair)==expected
cols=['split','stock','as_of_date','input_start_date','input_end_date','label_start_date','label_end_date']
meta=pd.read_csv(pair,usecols=cols,dtype=str); validate_test(meta); meta['row_id']=np.arange(len(meta))
y=np.load(cache/'y.npy',mmap_mode='r'); assert y.shape==(len(meta),10)
test=meta[meta.split.eq('test')]; ref=pd.read_csv(root/'experiments/csi300_aligned/models/prediction_adapter/predictions.csv',dtype={'stock':str})
a=test.merge(ref,on=['as_of_date','stock'],validate='one_to_one'); pc=[f'actual_return_{i}' for i in range(1,11)]
error=float(np.abs(y[a.row_id.to_numpy()]-a[pc].to_numpy()).max()); assert error<=1e-7,error
path=cache/'meta.csv'; assert not path.exists(); meta.to_csv(path,index=False)
report={'status':'recovered_from_hash_verified_original_pairs','rows':len(meta),'test_rows':len(test),'pairs_sha256':expected,'meta_sha256':sha256(path),'test_labels_max_error':error,'procedure':'exact metadata read/usecols/order and row_id assignment from src/doubleadapt/data.py build_cache'}
(root/'.runtime/hca_metadata_recovery.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
