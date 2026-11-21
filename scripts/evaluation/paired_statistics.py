"""Shared resampling and multiple-comparison utilities."""
import numpy as np


def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p,kind='stable');n=len(p)
    corrected=np.maximum.accumulate(p[order]*np.arange(n,0,-1)).clip(0,1)
    out=np.empty(n);out[order]=corrected
    return out


def block_indices(n,width,reps=5000,seed=100):
    width=min(n,width);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-width+1,size=(reps,(n+width-1)//width))
    return (starts[:,:,None]+np.arange(width)).reshape(reps,-1)[:,:n]
