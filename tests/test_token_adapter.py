import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/Kronos_official_src/Kronos-master'))
from model import Kronos
from src.kronos_continual.token_adapter import TokenPredictionAdapter, shifted_future_loss, normalize_history_future


def tiny():
    return Kronos(3,3,2,16,2,32,0.,0.,0.,0.,True).eval()


def test_identity_native_decode_and_frozen_gradients():
    torch.manual_seed(3); torch.set_num_threads(1)
    base=tiny(); model=TokenPredictionAdapter(base,bottleneck=4)
    model.train(); assert not base.training and model.adapter.training
    ids=torch.randint(0,8,(3,8)); ids2=torch.randint(0,8,(3,8))
    raw,h=base.decode_s1(ids,ids2)
    actual,z=model.decode_s1(ids,ids2)
    torch.testing.assert_close(actual,raw,rtol=0,atol=0)
    torch.testing.assert_close(z,h,rtol=0,atol=0)
    coarse=torch.randint(0,8,(3,1))
    torch.testing.assert_close(model.decode_s2(z,coarse),base.decode_s2(h,coarse),rtol=0,atol=0)
    frozen={k:v.clone() for k,v in base.state_dict().items()}
    out1,out2=model(ids[:,:-1],ids2[:,:-1])
    loss,*_=shifted_future_loss(out1,out2,ids,ids2,5)
    loss.backward()
    assert model.adapter.up.weight.grad.norm()>0
    assert all(p.grad is None for p in base.parameters())
    for k,v in frozen.items():torch.testing.assert_close(v,base.state_dict()[k],rtol=0,atol=0)


def test_parallel_fine_is_causal_in_eval_and_matches_last_generation_query():
    torch.manual_seed(4);base=tiny();model=TokenPredictionAdapter(base,4).eval()
    context=torch.randn(2,7,16);coarse=torch.randint(0,8,(2,7))
    expected=model._causal_fine_logits(context,coarse)
    poisoned=context.clone();poisoned[:,4:]=torch.randn_like(poisoned[:,4:])*100
    changed=model._causal_fine_logits(poisoned,coarse)
    torch.testing.assert_close(changed[:,:4],expected[:,:4],rtol=0,atol=1e-6)
    # Sequential training and generation use the identical native dependency weights.
    for i in range(7):
        native=model.decode_s2(context[:,:i+1],coarse[:,i:i+1])[:,-1]
        torch.testing.assert_close(expected[:,i],native,rtol=1e-5,atol=1e-5)


def test_future_only_loss_alignment():
    torch.manual_seed(2)
    t1=torch.randint(0,8,(2,10));t2=torch.randint(0,8,(2,10))
    l1=torch.randn(2,9,8,requires_grad=True);l2=torch.randn(2,9,8,requires_grad=True)
    loss,*_=shifted_future_loss(l1,l2,t1,t2,6);loss.backward()
    assert l1.grad[:,:5].abs().sum()==0
    assert l1.grad[:,5:].abs().sum()>0
    torch.testing.assert_close(loss,(torch.nn.functional.cross_entropy(l1[:,5:].reshape(-1,8),t1[:,6:].reshape(-1))+torch.nn.functional.cross_entropy(l2[:,5:].reshape(-1,8),t2[:,6:].reshape(-1)))/2)


def test_history_statistics_and_future_tokenization_causality():
    h=np.arange(24,dtype='float32').reshape(4,6);f=h[-2:]+3
    full,mean,scale=normalize_history_future(h,f)
    altered,mean2,scale2=normalize_history_future(h,f+999)
    np.testing.assert_array_equal(full[:4],altered[:4]);np.testing.assert_array_equal(mean,mean2)
    np.testing.assert_array_equal(scale,scale2)


def test_return_checkpoint_loading_is_explicit(tmp_path):
    model=TokenPredictionAdapter(tiny(),4)
    state={'adapter.'+k:v.clone() for k,v in model.adapter.state_dict().items()}
    state['head.weight']=torch.randn(10,16);path=tmp_path/'return.pt';torch.save(state,path)
    model.load_return_adapter(path)
    clone=TokenPredictionAdapter(tiny(),4);clone.load_state_dict(model.state_dict())
    assert set(model.state_dict())==set(clone.state_dict())
