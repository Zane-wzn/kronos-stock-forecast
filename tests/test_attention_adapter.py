import copy
import io

import pytest
import torch

from src.kronos_continual.attention_adapter import TemporalAttentionModel, TemporalAttentionPool


@pytest.mark.parametrize('batch,time', [(1,1),(3,5),(16,90)])
def test_shape_initial_identity_and_roundtrip(batch,time):
    torch.manual_seed(7)
    seq=torch.randn(batch,time,512)
    for mode in ('temporal_mean','temporal_attention'):
        model=TemporalAttentionModel(mode=mode).eval()
        expected=model.head(model.adapter(seq[:,-1]))
        torch.testing.assert_close(model(seq),expected,rtol=0,atol=0)
        buf=io.BytesIO(); torch.save(model.state_dict(),buf); buf.seek(0)
        restored=TemporalAttentionModel(mode=mode).eval()
        restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(model(seq),restored(seq),rtol=0,atol=0)


def test_mask_excludes_keys_and_preserves_full_history():
    torch.manual_seed(3)
    x=torch.randn(2,5,8)
    mask=torch.tensor([[True,False,True,False,False],[False]*5])
    for uniform in (False,True):
        pool=TemporalAttentionPool(8,4,2,uniform=uniform)
        torch.nn.init.normal_(pool.out_proj.weight)
        out,w=pool(x,mask,return_weights=True)
        assert (w[mask]==0).all()
        assert (w[~mask]>0).all()
        torch.testing.assert_close(w.sum(1),torch.ones(2))
        modified=x.clone(); modified[mask]=100
        torch.testing.assert_close(pool(modified,mask),out)
        with pytest.raises(ValueError): pool(x,torch.ones_like(mask))
        with pytest.raises(ValueError): pool(x,torch.ones(2,5))
        with pytest.raises(ValueError): pool(x.clone().fill_(float('nan')))


def test_gradient_starts_at_output_then_reaches_qkv():
    torch.manual_seed(9)
    x=torch.randn(16,90,512)
    target=torch.randn(16,10)
    model=TemporalAttentionModel().eval()
    # Mimic the already-trained Adapter in the actual experiment.
    torch.nn.init.normal_(model.adapter.up.weight,std=.01)
    ((model(x)-target)**2).mean().backward()
    assert model.pool.out_proj.weight.grad.norm()>0
    assert model.pool.q_proj.weight.grad.norm()==0
    with torch.no_grad(): model.pool.out_proj.weight.add_(-.01*model.pool.out_proj.weight.grad)
    model.zero_grad(); ((model(x)-target)**2).mean().backward()
    for part in (model.pool.q_proj,model.pool.k_proj,model.pool.v_proj,model.adapter.down,model.head):
        assert part.weight.grad is not None and part.weight.grad.norm()>0


def test_uniform_equals_explicit_value_average():
    x=torch.randn(2,9,512)
    model=TemporalAttentionModel(mode='temporal_mean').eval()
    torch.nn.init.normal_(model.pool.out_proj.weight)
    mean=model.pool.v_proj(model.pool.norm(x)).mean(1)
    expected=model.head(model.adapter(x[:,-1]+model.pool.out_proj(mean)))
    torch.testing.assert_close(model(x),expected)
